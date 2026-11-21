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

package org.graphframes.dataset.lib

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.col
import org.apache.spark.storage.StorageLevel
import org.graphframes.Logging
import org.graphframes.WithCheckpointInterval
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithLocalCheckpoints
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphEdge
import org.graphframes.dataset.GraphTriplet
import org.graphframes.dataset.GraphVertex
import org.graphframes.dataset.VertexId

import java.io.IOException
import scala.reflect.runtime.universe.TypeTag

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

private object AggregateNeighborsDatasetCols {
  val SRC_ID_COL = "srcId"
  val DST_ID_COL = "dstId"
  val SRC_ATTR_COL = "srcAttr"
  val DST_ATTR_COL = "dstAttr"
  val EDGE_ATTR_COL = "edgeAttr"
  val ACC_COL = "acc"
  val SHOULD_STOP_COL = "shouldStop"
  val IS_TARGET_COL = "isTarget"
}

/**
 * Typed multi-hop neighbor traversal and accumulation for
 * [[org.graphframes.dataset.GraphDataset]].
 *
 * Mirrors [[org.graphframes.lib.AggregateNeighbors]] with compile-time type safety: configure
 * starting vertices, `maxHops`, optional edge filtering, and then call [[withAccumulator]] to
 * define the typed accumulator state `S`, stopping/target conditions, and run the traversal.
 */
class AggregateNeighborsDataset[VD: TypeTag, ED: TypeTag] private[dataset] (
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

    import AggregateNeighborsDatasetCols.*
    val filteredEdges = {
      val base = edgePredicate.map(graph.edges.filter).getOrElse(graph.edges)
      if (removeLoops) base.filter(col(SRC_ID_COL) =!= col(DST_ID_COL)) else base
    }

    val semiTriplets = filteredEdges
      .toDF(SRC_ID_COL, DST_ID_COL, EDGE_ATTR_COL)
      .join(graph.vertices.toDF(DST_ID_COL, DST_ATTR_COL), Seq(DST_ID_COL), "inner")
      .repartition(col(SRC_ID_COL))
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
          .toDF(SRC_ID_COL, SRC_ATTR_COL, ACC_COL)
          .join(semiTriplets, Seq(SRC_ID_COL), "inner")
          .select(
            col(SRC_ID_COL),
            col(DST_ID_COL),
            col(SRC_ATTR_COL),
            col(DST_ATTR_COL),
            col(EDGE_ATTR_COL),
            col(ACC_COL))
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
          .filter(!col(SHOULD_STOP_COL))
          .map(t => TraversalFrontierState(t.id, t.attr, t.acc))

        val newlyCollected = {
          val filtered =
            if (hasTarget) transitions.filter(col(IS_TARGET_COL))
            else transitions.filter(col(SHOULD_STOP_COL))
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
