/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.graphframes.lib

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.struct
import org.apache.spark.sql.functions.typedLit
import org.apache.spark.storage.StorageLevel
import org.graphframes.GraphDataset
import org.graphframes.GraphFrame
import org.graphframes.Logging
import org.graphframes.WithCheckpointInterval
import org.graphframes.WithDirection
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithLocalCheckpoints

import java.io.IOException
import java.util
import scala.reflect.runtime.universe.TypeTag

/**
 * A typed graph path consisting of ordered vertices and the edges connecting them.
 *
 * For a 0-hop path (where a vertex matches both source and destination predicates), `vertices`
 * has 1 element and `edges` is empty (`length == 0`). For a `k`-hop path, `vertices.size == k +
 * 1` and `edges.size == k`.
 */
case class GraphPath[VD, ED](vertices: Seq[GraphVertex[VD]], edges: Seq[GraphEdge[ED]])
    extends Serializable {

  /** The starting vertex of the path. */
  def from: GraphVertex[VD] = vertices.head

  /** The ending vertex of the path. */
  def to: GraphVertex[VD] = vertices.last

  /** The number of edges in the path. */
  def length: Int = edges.size
}

object GraphPath {
  implicit def encoder[VD: TypeTag, ED: TypeTag]: Encoder[GraphPath[VD, ED]] =
    Encoders.product[GraphPath[VD, ED]]
}

private[lib] case class BFSFlaggedVertex[VD](
    id: VertexId,
    attr: VD,
    _gf_bfs_from: Boolean,
    _gf_bfs_to: Boolean)
    extends Serializable

private[lib] object BFSFlaggedVertex {
  implicit def encoder[VD: TypeTag]: Encoder[BFSFlaggedVertex[VD]] =
    Encoders.product[BFSFlaggedVertex[VD]]
}

/**
 * Typed Breadth-First Search (BFS) for [[org.graphframes.GraphDataset]].
 *
 * Finds all globally shortest paths from vertices matching `fromPred` to vertices matching
 * `toPred` up to `maxPathLength` hops, returning a typed `Dataset[GraphPath[VD, ED]]`.
 */
class BFSDataset[VD: TypeTag, ED: TypeTag] private[graphframes] (
    private val graph: GraphDataset[VD, ED])
    extends Arguments
    with Serializable {

  private var maxPathLen: Int = 10
  private var fromPredicate: Option[GraphVertex[VD] => Boolean] = None
  private var toPredicate: Option[GraphVertex[VD] => Boolean] = None
  private var edgePredicate: Option[GraphEdge[ED] => Boolean] = None

  /** Sets the typed predicate identifying starting vertices. */
  def fromPred(predicate: GraphVertex[VD] => Boolean): this.type = {
    fromPredicate = Some(predicate)
    this
  }

  /** Sets the typed predicate identifying target vertices. */
  def toPred(predicate: GraphVertex[VD] => Boolean): this.type = {
    toPredicate = Some(predicate)
    this
  }

  /** Sets an optional typed predicate restricting which edges may be traversed. */
  def edgeFilter(predicate: GraphEdge[ED] => Boolean): this.type = {
    edgePredicate = Some(predicate)
    this
  }

  /** Sets the maximum path length (number of edges, >= 0, default: 10). */
  def maxPathLength(value: Int): this.type = {
    require(value >= 0, s"BFS maxPathLength must be >= 0, but was set to $value")
    maxPathLen = value
    this
  }

  /** Executes BFS and returns a typed `Dataset[GraphPath[VD, ED]]`. */
  def run(): Dataset[GraphPath[VD, ED]] = {
    val fromFn =
      fromPredicate.getOrElse(throw new IllegalArgumentException("fromPred is required."))
    val toFn = toPredicate.getOrElse(throw new IllegalArgumentException("toPred is required."))

    val spark = graph.spark

    val fromCol = "_gf_bfs_from"
    val toCol = "_gf_bfs_to"

    val vDF = graph.vertices
      .map(v => BFSFlaggedVertex(v.id, v.attr, fromFn(v), toFn(v)))
      .toDF(GraphFrame.ID, GraphFrame.ATTR, fromCol, toCol)

    val filteredEdges = edgePredicate.map(graph.edges.filter).getOrElse(graph.edges)
    val eDF = filteredEdges.toDF(GraphFrame.SRC, GraphFrame.DST, GraphFrame.ATTR)

    val resDF = GraphFrame(vDF, eDF).bfs
      .fromExpr(col(fromCol))
      .toExpr(col(toCol))
      .maxPathLength(maxPathLen)
      .run()

    val cols = resDF.columns
    if (!cols.contains("from") || !cols.contains("to")) {
      return spark.emptyDataset[GraphPath[VD, ED]]
    }

    val edgeCols =
      cols.filter(c => c.startsWith("e") && c.length > 1 && c.drop(1).forall(_.isDigit))
    if (edgeCols.isEmpty) {
      // 0-hop paths: from == to
      resDF
        .select(
          array(struct(col("from.id").alias("id"), col("from.attr").alias("attr")))
            .alias("vertices"),
          typedLit(Seq.empty[GraphEdge[ED]]).alias("edges"))
        .as[GraphPath[VD, ED]]
    } else {
      val vertexCols = cols.filter(c =>
        c == "from" || c == "to" ||
          (c.startsWith("v") && c.length > 1 && c.drop(1).forall(_.isDigit)))
      val verticesExpr = array(
        vertexCols
          .map(c => struct(col(s"$c.id").alias("id"), col(s"$c.attr").alias("attr")))
          .toIndexedSeq: _*)
      val edgesExpr = array(
        edgeCols
          .map(c =>
            struct(
              col(s"$c.${GraphFrame.SRC}").alias("srcId"),
              col(s"$c.${GraphFrame.DST}").alias("dstId"),
              col(s"$c.${GraphFrame.ATTR}").alias("attr")))
          .toIndexedSeq: _*)

      resDF
        .select(verticesExpr.alias("vertices"), edgesExpr.alias("edges"))
        .as[GraphPath[VD, ED]]
    }
  }
}

/**
 * A simple path represented by ordered vertex IDs and its edge count.
 */
case class SimplePath(path: Seq[VertexId], length: Int) extends Serializable {

  /** The starting vertex ID. */
  def from: VertexId = path.head

  /** The destination vertex ID. */
  def to: VertexId = path.last

  /** Returns the ordered vertex IDs in this path as an unmodifiable Java {@code List}. */
  def pathAsJava: util.List[java.lang.Long] = {
    val list = new util.ArrayList[java.lang.Long](path.size)
    path.foreach { id =>
      val _ = list.add(java.lang.Long.valueOf(id))
    }
    util.Collections.unmodifiableList(list)
  }
}

object SimplePath {
  implicit val encoder: Encoder[SimplePath] = Encoders.product[SimplePath]
}

/**
 * Typed AllPaths algorithm for [[org.graphframes.GraphDataset]], wrapping [[AllPaths]].
 *
 * Enumerates all simple paths (no repeated vertices) up to `maxPathLength` hops between vertices
 * matching `fromPred` and vertices matching `toPred`.
 */
class AllPathsDataset[VD, ED] private[graphframes] (private val graph: GraphDataset[VD, ED])
    extends Arguments
    with Serializable
    with WithDirection
    with WithLocalCheckpoints
    with WithCheckpointInterval
    with WithIntermediateStorageLevel {

  private var maxPathLen: Int = 5
  private var fromPredicate: Option[GraphVertex[VD] => Boolean] = None
  private var toPredicate: Option[GraphVertex[VD] => Boolean] = None
  private var edgePredicate: Option[GraphEdge[ED] => Boolean] = None

  /** Sets the typed predicate identifying source vertices. */
  def fromPred(predicate: GraphVertex[VD] => Boolean): this.type = {
    fromPredicate = Some(predicate)
    this
  }

  /** Sets the typed predicate identifying destination vertices. */
  def toPred(predicate: GraphVertex[VD] => Boolean): this.type = {
    toPredicate = Some(predicate)
    this
  }

  /** Sets an optional typed predicate restricting which edges may be traversed. */
  def edgeFilter(predicate: GraphEdge[ED] => Boolean): this.type = {
    edgePredicate = Some(predicate)
    this
  }

  /** Sets the maximum path length (number of edges, > 0, default: 5). */
  def maxPathLength(value: Int): this.type = {
    require(value > 0, s"AllPaths maxPathLength must be > 0, but was set to $value")
    maxPathLen = value
    this
  }

  /** Executes AllPaths and returns a typed `Dataset[SimplePath]`. */
  def run(): Dataset[SimplePath] = {
    val fromFn =
      fromPredicate.getOrElse(throw new IllegalArgumentException("fromPred is required."))
    val toFn = toPredicate.getOrElse(throw new IllegalArgumentException("toPred is required."))

    val spark = graph.spark
    import spark.implicits.*

    val fromCol = "_gf_ap_from"
    val toCol = "_gf_ap_to"

    val vDF = graph.vertices
      .map(v => (v.id, fromFn(v), toFn(v)))
      .toDF(GraphFrame.ID, fromCol, toCol)

    val filteredEdges = edgePredicate.map(graph.edges.filter).getOrElse(graph.edges)
    val eDF = filteredEdges
      .select(col("srcId").alias(GraphFrame.SRC), col("dstId").alias(GraphFrame.DST))
      .distinct()

    GraphFrame(vDF, eDF).allPaths
      .fromExpr(col(fromCol))
      .toExpr(col(toCol))
      .maxPathLength(maxPathLen)
      .setIsDirected(isDirected)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .setCheckpointInterval(checkpointInterval)
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .run()
      .select(col("path"), col("len").cast("int").alias("length"))
      .as[SimplePath]
  }
}

/**
 * Result row for [[AggregateNeighborsDataset]], containing the reached vertex ID, hop count, and
 * accumulated state `S`.
 */
case class NeighborAggregateResult[S](id: VertexId, hop: Int, accumulator: S) extends Serializable

object NeighborAggregateResult {
  implicit def encoder[S: TypeTag]: Encoder[NeighborAggregateResult[S]] =
    Encoders.product[NeighborAggregateResult[S]]
}

private[lib] case class TraversalFrontierState[VD, S](id: VertexId, attr: VD, acc: S)
    extends Serializable

private[lib] object TraversalFrontierState {
  implicit def encoder[VD: TypeTag, S: TypeTag]: Encoder[TraversalFrontierState[VD, S]] =
    Encoders.product[TraversalFrontierState[VD, S]]
}

private[lib] case class TraversalStep[VD, ED, S](
    srcId: VertexId,
    dstId: VertexId,
    srcAttr: VD,
    dstAttr: VD,
    edgeAttr: ED,
    acc: S)
    extends Serializable

private[lib] object TraversalStep {
  implicit def encoder[VD: TypeTag, ED: TypeTag, S: TypeTag]: Encoder[TraversalStep[VD, ED, S]] =
    Encoders.product[TraversalStep[VD, ED, S]]
}

private[lib] case class TraversalTransition[VD, S](
    id: VertexId,
    attr: VD,
    acc: S,
    shouldStop: Boolean,
    isTarget: Boolean)
    extends Serializable

private[lib] object TraversalTransition {
  implicit def encoder[VD: TypeTag, S: TypeTag]: Encoder[TraversalTransition[VD, S]] =
    Encoders.product[TraversalTransition[VD, S]]
}

/**
 * Typed multi-hop neighbor traversal and accumulation for [[org.graphframes.GraphDataset]].
 *
 * Mirrors [[AggregateNeighbors]] with compile-time type safety: configure starting vertices,
 * `maxHops`, optional edge filtering, and then call [[withAccumulator]] to define the typed
 * accumulator state `S`, stopping/target conditions, and run the traversal.
 */
class AggregateNeighborsDataset[VD: TypeTag, ED: TypeTag] private[graphframes] (
    private val graph: GraphDataset[VD, ED])
    extends Serializable
    with Logging
    with WithIntermediateStorageLevel
    with WithCheckpointInterval
    with WithLocalCheckpoints {

  private var startingPredicate: GraphVertex[VD] => Boolean = (_: GraphVertex[VD]) => true
  private var maxHops: Int = 3
  private var edgePredicate: Option[GraphEdge[ED] => Boolean] = None
  private var removeLoops: Boolean = false

  /** Sets the predicate selecting seed vertices for the traversal (default: all vertices). */
  def setStartingVertices(predicate: GraphVertex[VD] => Boolean): this.type = {
    startingPredicate = predicate
    this
  }

  /** Sets the maximum number of hops to explore (must be > 0, default: 3). */
  def setMaxHops(value: Int): this.type = {
    require(value > 0, "maxHops should be positive.")
    maxHops = value
    this
  }

  /** Sets an optional predicate filtering which edges may be traversed. */
  def setEdgeFilter(predicate: GraphEdge[ED] => Boolean): this.type = {
    edgePredicate = Some(predicate)
    this
  }

  /**
   * Sets whether self-loop edges (`srcId == dstId`) are removed before traversal (default:
   * false).
   */
  def setRemoveLoops(value: Boolean): this.type = {
    removeLoops = value
    this
  }

  /**
   * Configures the typed accumulator state `S` with an initializer (evaluated on starting
   * vertices) and an update function (evaluated on `(previousAccumulator, traversedTriplet)` at
   * each hop).
   */
  def withAccumulator[S: TypeTag](
      init: GraphVertex[VD] => S,
      update: (S, GraphTriplet[VD, ED]) => S): TypedAggregateNeighborsBuilder[VD, ED, S] = {
    new TypedAggregateNeighborsBuilder[VD, ED, S](
      graph = graph,
      startingPredicate = startingPredicate,
      maxHops = maxHops,
      edgePredicate = edgePredicate,
      removeLoops = removeLoops,
      checkpointInterval = checkpointInterval,
      useLocalCheckpoints = useLocalCheckpoints,
      intermediateStorageLevel = intermediateStorageLevel,
      init = init,
      update = update)
  }
}

/**
 * Fluent typed builder for executing an [[AggregateNeighborsDataset]] traversal with accumulator
 * type `S`.
 */
class TypedAggregateNeighborsBuilder[VD: TypeTag, ED: TypeTag, S: TypeTag] private[lib] (
    private val graph: GraphDataset[VD, ED],
    private val startingPredicate: GraphVertex[VD] => Boolean,
    private val maxHops: Int,
    private val edgePredicate: Option[GraphEdge[ED] => Boolean],
    private val removeLoops: Boolean,
    private val checkpointInterval: Int,
    private val useLocalCheckpoints: Boolean,
    private val intermediateStorageLevel: StorageLevel,
    private val init: GraphVertex[VD] => S,
    private val update: (S, GraphTriplet[VD, ED]) => S)
    extends Serializable
    with Logging {

  private var stoppingCond: Option[(S, GraphTriplet[VD, ED]) => Boolean] = None
  private var targetCond: Option[(S, GraphTriplet[VD, ED]) => Boolean] = None

  /**
   * Sets the stopping condition evaluated on `(previousAccumulator, traversedTriplet)`. When
   * `true`, exploration along that path stops. If no `targetCondition` is set, stopped paths are
   * collected into the result.
   */
  def setStoppingCondition(predicate: (S, GraphTriplet[VD, ED]) => Boolean): this.type = {
    stoppingCond = Some(predicate)
    this
  }

  /**
   * Sets the target condition evaluated on `(previousAccumulator, traversedTriplet)`. When
   * `true`, the updated accumulator is collected into the result and exploration along that path
   * stops.
   */
  def setTargetCondition(predicate: (S, GraphTriplet[VD, ED]) => Boolean): this.type = {
    targetCond = Some(predicate)
    this
  }

  /**
   * Executes the breadth-first neighbor aggregation and returns a persisted and materialized
   * `Dataset[NeighborAggregateResult[S]]`. Callers should call `.unpersist()` on the returned
   * dataset when it is no longer needed.
   */
  def run(): Dataset[NeighborAggregateResult[S]] = {
    require(
      stoppingCond.isDefined || targetCond.isDefined,
      "At least one of setStoppingCondition or setTargetCondition must be set.")

    val spark = graph.spark
    if (checkpointInterval > 0 && !useLocalCheckpoints && spark.sparkContext.getCheckpointDir.isEmpty) {
      spark.conf.getOption("spark.checkpoint.dir") match {
        case Some(d) => spark.sparkContext.setCheckpointDir(d)
        case None =>
          throw new IOException(
            "Checkpoint directory is not set. Please set it first using sc.setCheckpointDir() " +
              "or enable local checkpoints via setUseLocalCheckpoints(true).")
      }
    }

    val filteredEdges = {
      val base = edgePredicate.map(graph.edges.filter).getOrElse(graph.edges)
      if (removeLoops) base.filter(col("srcId") =!= col("dstId")) else base
    }

    val semiTriplets = filteredEdges
      .toDF("srcId", "dstId", "edgeAttr")
      .join(graph.vertices.toDF("dstId", "dstAttr"), Seq("dstId"), "inner")
      .repartition(col("srcId"))
      .persist(intermediateStorageLevel)

    val initFn = init
    val stopFnOpt = stoppingCond
    val targetFnOpt = targetCond
    val updateFn = update
    val hasTarget = targetFnOpt.isDefined

    var states: Dataset[TraversalFrontierState[VD, S]] = null
    var finished: Dataset[NeighborAggregateResult[S]] = null
    var prevStates: Dataset[TraversalFrontierState[VD, S]] = null
    var prevFinished: Dataset[NeighborAggregateResult[S]] = null
    var transitions: Dataset[TraversalTransition[VD, S]] = null
    var succeeded = false

    try {
      states = graph.vertices
        .filter(startingPredicate)
        .map(v => TraversalFrontierState(v.id, v.attr, initFn(v)))
        .persist(intermediateStorageLevel)

      finished = spark.emptyDataset[NeighborAggregateResult[S]].persist(intermediateStorageLevel)

      var iter = 0
      var converged = states.count() == 0

      while (!converged && iter < maxHops) {
        iter += 1
        val currentHop = iter
        prevStates = states
        prevFinished = finished

        transitions = prevStates
          .toDF("srcId", "srcAttr", "acc")
          .join(semiTriplets, Seq("srcId"), "inner")
          .select(
            col("srcId"),
            col("dstId"),
            col("srcAttr"),
            col("dstAttr"),
            col("edgeAttr"),
            col("acc"))
          .as[TraversalStep[VD, ED, S]]
          .map { step =>
            val triplet =
              GraphTriplet(step.srcId, step.dstId, step.srcAttr, step.dstAttr, step.edgeAttr)
            val newAcc = updateFn(step.acc, triplet)
            val isTarget = targetFnOpt.exists(fn => fn(step.acc, triplet))
            val shouldStop = stopFnOpt.exists(fn => fn(step.acc, triplet)) || isTarget
            TraversalTransition(step.dstId, step.dstAttr, newAcc, shouldStop, isTarget)
          }
          .persist(intermediateStorageLevel)

        val nextStatesRaw = transitions
          .filter(!col("shouldStop"))
          .map(t => TraversalFrontierState(t.id, t.attr, t.acc))

        val newlyCollected = {
          val filtered =
            if (hasTarget) transitions.filter(col("isTarget"))
            else transitions.filter(col("shouldStop"))
          filtered.map(t => NeighborAggregateResult(t.id, currentHop, t.acc))
        }

        val nextFinishedRaw = prevFinished.union(newlyCollected)

        val (maybeCkptStates, maybeCkptFinished) =
          if (checkpointInterval > 0 && iter % checkpointInterval == 0) {
            if (useLocalCheckpoints) {
              (
                nextStatesRaw.localCheckpoint(eager = false),
                nextFinishedRaw.localCheckpoint(eager = false))
            } else {
              (nextStatesRaw.checkpoint(eager = false), nextFinishedRaw.checkpoint(eager = false))
            }
          } else {
            (nextStatesRaw, nextFinishedRaw)
          }

        states = maybeCkptStates.persist(intermediateStorageLevel)
        finished = maybeCkptFinished.persist(intermediateStorageLevel)

        converged = {
          val _ = finished.count()
          states.count() == 0
        }

        val _ = (transitions.unpersist(), prevStates.unpersist(), prevFinished.unpersist())
        transitions = null
        prevStates = null
        prevFinished = null
      }

      if (iter == 0) {
        val _ = finished.count()
      }
      succeeded = true
      resultIsPersistent()
      finished
    } finally {
      if (!succeeded && finished != null) {
        val _ = finished.unpersist()
      }
      if (prevFinished != null) {
        val _ = prevFinished.unpersist()
      }
      if (transitions != null) {
        val _ = transitions.unpersist()
      }
      if (prevStates != null) {
        val _ = prevStates.unpersist()
      }
      if (states != null) {
        val _ = states.unpersist()
      }
      semiTriplets.unpersist()
      ()
    }
  }
}
