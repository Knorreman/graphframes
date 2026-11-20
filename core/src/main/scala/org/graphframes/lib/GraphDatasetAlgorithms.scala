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
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.struct
import org.graphframes.GraphDataset
import org.graphframes.GraphFrame
import org.graphframes.Logging
import org.graphframes.WithAlgorithmChoice
import org.graphframes.WithCheckpointInterval
import org.graphframes.WithDirection
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithLgNomEntries
import org.graphframes.WithLocalCheckpoints
import org.graphframes.WithMaxIter

import scala.reflect.runtime.universe.TypeTag

/** Vertex attribute augmented with PageRank score. */
case class PageRankAttr[VD](attr: VD, pagerank: Double) extends Serializable

object PageRankAttr {
  implicit def encoder[VD: TypeTag]: Encoder[PageRankAttr[VD]] =
    Encoders.product[PageRankAttr[VD]]
}

/** Edge attribute augmented with normalized weight (e.g., from PageRank). */
case class WeightedEdgeAttr[ED](attr: ED, weight: Double) extends Serializable

object WeightedEdgeAttr {
  implicit def encoder[ED: TypeTag]: Encoder[WeightedEdgeAttr[ED]] =
    Encoders.product[WeightedEdgeAttr[ED]]
}

/**
 * PageRank algorithm for [[org.graphframes.GraphDataset]], wrapping [[PageRank]].
 */
class PageRankDataset[VD: TypeTag, ED: TypeTag] private[graphframes] (
    private val graph: GraphDataset[VD, ED])
    extends Arguments
    with Logging {

  private var tolOpt: Option[Double] = None
  private var resetProb: Double = 0.15
  private var maxIterOpt: Option[Int] = None
  private var srcIdOpt: Option[VertexId] = None

  /** Source vertex for Personalized PageRank (optional). */
  def sourceId(value: VertexId): this.type = {
    srcIdOpt = Some(value)
    this
  }

  /** Reset probability "alpha" (default: 0.15). */
  def resetProbability(value: Double): this.type = {
    resetProb = value
    this
  }

  /** Convergence tolerance (mutually exclusive with [[maxIter]]). */
  def tol(value: Double): this.type = {
    tolOpt = Some(value)
    this
  }

  /** Fixed number of iterations (mutually exclusive with [[tol]]). */
  def maxIter(value: Int): this.type = {
    maxIterOpt = Some(value)
    this
  }

  /**
   * Runs PageRank and returns a [[GraphDataset]] whose vertex attributes are [[PageRankAttr]] and
   * edge attributes are [[WeightedEdgeAttr]].
   */
  def run(): GraphDataset[PageRankAttr[VD], WeightedEdgeAttr[ED]] = {
    val pr = graph.toGraphFrame.pageRank.resetProbability(resetProb)
    srcIdOpt.foreach(pr.sourceId)
    tolOpt.foreach(pr.tol)
    maxIterOpt.foreach(pr.maxIter)

    val resGF = pr.run()
    val vv = resGF.vertices
      .select(
        col(GraphFrame.ID).alias("id"),
        struct(col(GraphFrame.ATTR).alias("attr"), col("pagerank").alias("pagerank"))
          .alias("attr"))
      .as[GraphVertex[PageRankAttr[VD]]]
    val ee = resGF.edges
      .select(
        col(GraphFrame.SRC).alias("srcId"),
        col(GraphFrame.DST).alias("dstId"),
        struct(col(GraphFrame.ATTR).alias("attr"), col("weight").alias("weight")).alias("attr"))
      .as[GraphEdge[WeightedEdgeAttr[ED]]]
    GraphDataset(vv, ee)
  }
}

/** Result row for [[ShortestPathsDataset]], mapping each vertex to landmark distances. */
case class ShortestPathsResult(id: VertexId, distances: Map[VertexId, Int])
    extends GraphVertexBase[Map[VertexId, Int]] {
  def attr: Map[VertexId, Int] = distances
  def toVertex: GraphVertex[Map[VertexId, Int]] = GraphVertex(id, distances)
}

object ShortestPathsResult {
  implicit val encoder: Encoder[ShortestPathsResult] =
    Encoders.product[ShortestPathsResult]
}

/**
 * Shortest Paths algorithm for [[org.graphframes.GraphDataset]], wrapping [[ShortestPaths]].
 */
class ShortestPathsDataset[VD, ED] private[graphframes] (private val graph: GraphDataset[VD, ED])
    extends Arguments
    with WithAlgorithmChoice
    with WithCheckpointInterval
    with WithLocalCheckpoints
    with WithIntermediateStorageLevel
    with WithDirection
    with Logging {

  private var lmarks: Option[Seq[VertexId]] = None

  /** The list of landmark vertex IDs to compute shortest paths to. */
  def landmarks(value: Seq[VertexId]): this.type = {
    lmarks = Some(value)
    this
  }

  /** Runs shortest paths and returns a typed dataset of [[ShortestPathsResult]]. */
  def run(): Dataset[ShortestPathsResult] = {
    val lmarksChecked = check(lmarks, "landmarks")
    val sp = graph.toGraphFrame.shortestPaths
      .landmarks(lmarksChecked)
      .setAlgorithm(algorithm)
      .setCheckpointInterval(checkpointInterval)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setIsDirected(isDirected)

    sp.run()
      .select(col(GraphFrame.ID).alias("id"), col("distances"))
      .as[ShortestPathsResult]
  }
}

/** Result row for [[TriangleCountDataset]]. */
case class TriangleCountResult(id: VertexId, count: Long) extends GraphVertexBase[Long] {
  def attr: Long = count
  def toVertex: GraphVertex[Long] = GraphVertex(id, count)
}

object TriangleCountResult {
  implicit val encoder: Encoder[TriangleCountResult] =
    Encoders.product[TriangleCountResult]
}

/**
 * Triangle Count algorithm for [[org.graphframes.GraphDataset]], wrapping [[TriangleCount]].
 */
class TriangleCountDataset[VD, ED] private[graphframes] (private val graph: GraphDataset[VD, ED])
    extends Arguments
    with Serializable
    with WithIntermediateStorageLevel
    with WithLgNomEntries {

  private var algorithm: String = "exact"

  /** Sets the triangle counting algorithm: `"exact"` (default) or `"approx"`. */
  def setAlgorithm(value: String): this.type = {
    algorithm = value
    this
  }

  /** Runs triangle counting and returns a typed dataset of [[TriangleCountResult]]. */
  def run(): Dataset[TriangleCountResult] = {
    val tc = graph.toGraphFrame.triangleCount
      .setAlgorithm(algorithm)
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setLgNomEntries(lgNomEntries)

    tc.run()
      .select(col(GraphFrame.ID).alias("id"), col("count").cast("long").alias("count"))
      .as[TriangleCountResult]
  }
}

/** Result row for [[LabelPropagationDataset]]. */
case class LabelPropagationResult(id: VertexId, label: Long) extends GraphVertexBase[Long] {
  def attr: Long = label
  def toVertex: GraphVertex[Long] = GraphVertex(id, label)
}

object LabelPropagationResult {
  implicit val encoder: Encoder[LabelPropagationResult] =
    Encoders.product[LabelPropagationResult]
}

/**
 * Label Propagation algorithm for [[org.graphframes.GraphDataset]], wrapping
 * [[LabelPropagation]].
 */
class LabelPropagationDataset[VD, ED] private[graphframes] (
    private val graph: GraphDataset[VD, ED])
    extends Arguments
    with WithAlgorithmChoice
    with WithCheckpointInterval
    with WithMaxIter
    with WithLocalCheckpoints
    with WithIntermediateStorageLevel
    with Logging {

  /** Runs label propagation and returns a typed dataset of [[LabelPropagationResult]]. */
  def run(): Dataset[LabelPropagationResult] = {
    val maxIterChecked = check(maxIter, "maxIter")
    val lp = graph.toGraphFrame.labelPropagation
      .maxIter(maxIterChecked)
      .setAlgorithm(algorithm)
      .setCheckpointInterval(checkpointInterval)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .setIntermediateStorageLevel(intermediateStorageLevel)

    lp.run()
      .select(col(GraphFrame.ID).alias("id"), col("label").cast("long").alias("label"))
      .as[LabelPropagationResult]
  }
}

/**
 * Strongly Connected Components algorithm for [[org.graphframes.GraphDataset]], wrapping
 * [[StronglyConnectedComponents]].
 */
class StronglyConnectedComponentsDataset[VD, ED] private[graphframes] (
    private val graph: GraphDataset[VD, ED])
    extends Arguments
    with WithMaxIter
    with Logging {

  /**
   * Runs strongly connected components and returns a typed dataset of
   * [[ConnectedComponentResult]].
   */
  def run(): Dataset[ConnectedComponentResult] = {
    val maxIterChecked = check(maxIter, "maxIter")
    graph.toGraphFrame.stronglyConnectedComponents
      .maxIter(maxIterChecked)
      .run()
      .select(col(GraphFrame.ID).alias("id"), col("component").alias("connectedComponent"))
      .as[ConnectedComponentResult]
  }
}

/** Result row for [[KCoreDataset]]. */
case class KCoreResult(id: VertexId, kcore: Int) extends GraphVertexBase[Int] {
  def attr: Int = kcore
  def toVertex: GraphVertex[Int] = GraphVertex(id, kcore)
}

object KCoreResult {
  implicit val encoder: Encoder[KCoreResult] =
    Encoders.product[KCoreResult]
}

/**
 * K-Core decomposition algorithm for [[org.graphframes.GraphDataset]], wrapping [[KCore]].
 */
class KCoreDataset[VD, ED] private[graphframes] (private val graph: GraphDataset[VD, ED])
    extends Serializable
    with WithIntermediateStorageLevel
    with WithCheckpointInterval
    with WithLocalCheckpoints {

  /** Runs K-Core decomposition and returns a typed dataset of [[KCoreResult]]. */
  def run(): Dataset[KCoreResult] = {
    graph.toGraphFrame.kCore
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setCheckpointInterval(checkpointInterval)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .run()
      .select(col(GraphFrame.ID).alias("id"), col(KCore.kCoreColumnName).alias("kcore"))
      .as[KCoreResult]
  }
}

/**
 * Maximal Independent Set algorithm for [[org.graphframes.GraphDataset]], wrapping
 * [[MaximalIndependentSet]].
 */
class MaximalIndependentSetDataset[VD: TypeTag, ED] private[graphframes] (
    private val graph: GraphDataset[VD, ED])
    extends Serializable
    with WithIntermediateStorageLevel
    with WithCheckpointInterval
    with WithLocalCheckpoints {

  /**
   * Runs Maximal Independent Set and returns the typed `Dataset[GraphVertex[VD]]` of vertices in
   * the independent set.
   */
  def run(seed: Long): Dataset[GraphVertex[VD]] = {
    val misIds = graph.toGraphFrame.maximalIndependentSet
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setCheckpointInterval(checkpointInterval)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .run(seed)
      .select(col(GraphFrame.ID).alias("id"))
    graph.vertices
      .join(misIds, Seq("id"), "left_semi")
      .as[GraphVertex[VD]]
  }
}

/**
 * Cycle detection algorithm for [[org.graphframes.GraphDataset]], wrapping [[DetectingCycles]].
 */
class DetectingCyclesDataset[VD, ED] private[graphframes] (
    private val graph: GraphDataset[VD, ED])
    extends Arguments
    with Serializable
    with Logging
    with WithIntermediateStorageLevel
    with WithLocalCheckpoints
    with WithCheckpointInterval {

  /** Runs cycle detection and returns a typed `Dataset[Seq[VertexId]]` of unique cycles. */
  def run(): Dataset[Seq[VertexId]] = {
    val session = graph.spark
    import session.implicits.*
    graph.toGraphFrame.detectingCycles
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .setCheckpointInterval(checkpointInterval)
      .run()
      .select(col(DetectingCycles.foundSeqCol))
      .as[Seq[VertexId]]
  }
}
