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

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.functions.coalesce
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.collect_list
import org.apache.spark.sql.functions.count
import org.apache.spark.sql.functions.map_from_entries
import org.apache.spark.sql.functions.struct
import org.apache.spark.sql.functions.when
import org.apache.spark.storage.StorageLevel
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
import org.graphframes.embeddings.Hash2Vec
import org.graphframes.embeddings.RandomWalkEmbeddings
import org.graphframes.rw.RandomWalkBase

import java.util
import scala.reflect.runtime.universe.TypeTag

private object AlgorithmCols {
  val ID_COL = "id"
  val ATTR_COL = "attr"
  val SRC_ID_COL = "srcId"
  val DST_ID_COL = "dstId"
  val PAGERANK_COL = "pagerank"
  val PAGERANKS_COL = "pageranks"
  val WEIGHT_COL = "weight"
  val DISTANCES_COL = "distances"
  val COUNT_COL = "count"
  val LABEL_COL = "label"
  val COMPONENT_COL = "component"
  val CONNECTED_COMPONENT_COL = "connectedComponent"
  val KCORE_COL = "kcore"
  val INIT_LABEL_COL = "_gf_init_label"
  val HOP_SKETCHES_COL = "hopSketches"
  val EMBEDDING_COL = "embedding"
  val KEY_COL = "k"
  val CNT_COL = "cnt"
  val DEGREES_COL = "degrees"
  val COLUMN1_COL = "column1"
  val COLUMN2_COL = "column2"
  val COLUMN3_COL = "column3"
  val COLUMN4_COL = "column4"
}

private[graphframes] object DatasetCacheUtils {

  /**
   * Persists and materializes a typed `Dataset` projected from a wrapped `GraphFrame` algorithm
   * result `DataFrame`, then unpersists `raw` so the intermediate DataFrame does not leak in
   * Spark's cache manager.
   */
  def materializeAndReleaseRaw[T](
      raw: DataFrame,
      typed: Dataset[T],
      storageLevel: StorageLevel): Dataset[T] = {
    if (typed.storageLevel != StorageLevel.NONE) {
      val _ = typed.count()
      typed
    } else {
      val persisted = typed.persist(storageLevel)
      var succeeded = false
      try {
        val _ = persisted.count()
        succeeded = true
        persisted
      } finally {
        if (!succeeded) {
          val _ = persisted.unpersist()
        }
        if (raw.storageLevel != StorageLevel.NONE) {
          val _ = raw.unpersist()
        }
        ()
      }
    }
  }

  /**
   * Persists and materializes both vertex and edge datasets projected from a wrapped `GraphFrame`
   * algorithm result, then unpersists `rawGF` so neither raw DataFrame leaks in Spark's cache
   * manager.
   */
  def materializeGraphAndReleaseRaw[VD: TypeTag, ED: TypeTag](
      rawGF: GraphFrame,
      typedVertices: Dataset[GraphVertex[VD]],
      typedEdges: Dataset[GraphEdge[ED]],
      storageLevel: StorageLevel): GraphDataset[VD, ED] = {
    val verticesAlreadyCached = typedVertices.storageLevel != StorageLevel.NONE
    val edgesAlreadyCached = typedEdges.storageLevel != StorageLevel.NONE
    val persistedV =
      if (verticesAlreadyCached) typedVertices else typedVertices.persist(storageLevel)
    val persistedE =
      if (edgesAlreadyCached) typedEdges else typedEdges.persist(storageLevel)
    var succeeded = false
    try {
      val _ = (persistedV.count(), persistedE.count())
      succeeded = true
      GraphDataset(persistedV, persistedE)
    } finally {
      if (!succeeded) {
        if (!verticesAlreadyCached) {
          val _ = persistedV.unpersist()
        }
        if (!edgesAlreadyCached) {
          val _ = persistedE.unpersist()
        }
      }
      if (!verticesAlreadyCached && rawGF.vertices.storageLevel != StorageLevel.NONE) {
        val _ = rawGF.vertices.unpersist()
      }
      if (!edgesAlreadyCached && rawGF.edges.storageLevel != StorageLevel.NONE) {
        val _ = rawGF.edges.unpersist()
      }
      ()
    }
  }
}

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
   * Runs PageRank and returns a persisted and materialized [[GraphDataset]] whose vertex
   * attributes are [[PageRankAttr]] and edge attributes are [[WeightedEdgeAttr]]. Callers should
   * call `.unpersist()` on the returned graph when it is no longer needed.
   */
  def run(): GraphDataset[PageRankAttr[VD], WeightedEdgeAttr[ED]] = {
    val pr = graph.toGraphFrame.pageRank.resetProbability(resetProb)
    srcIdOpt.foreach(pr.sourceId)
    tolOpt.foreach(pr.tol)
    maxIterOpt.foreach(pr.maxIter)

    import AlgorithmCols.*
    val resGF = pr.run()
    val vv = resGF.vertices
      .select(
        col(GraphFrame.ID).alias(ID_COL),
        struct(col(GraphFrame.ATTR).alias(ATTR_COL), col(PAGERANK_COL).alias(PAGERANK_COL))
          .alias(ATTR_COL))
      .as[GraphVertex[PageRankAttr[VD]]]
    val ee = resGF.edges
      .select(
        col(GraphFrame.SRC).alias(SRC_ID_COL),
        col(GraphFrame.DST).alias(DST_ID_COL),
        struct(col(GraphFrame.ATTR).alias(ATTR_COL), col(WEIGHT_COL).alias(WEIGHT_COL))
          .alias(ATTR_COL))
      .as[GraphEdge[WeightedEdgeAttr[ED]]]
    val res =
      DatasetCacheUtils.materializeGraphAndReleaseRaw(resGF, vv, ee, StorageLevel.MEMORY_AND_DISK)
    resultIsPersistent()
    res
  }
}

/** Result row for [[ShortestPathsDataset]], mapping each vertex to landmark distances. */
case class ShortestPathsResult(id: VertexId, distances: Map[VertexId, Int])
    extends GraphVertexBase[Map[VertexId, Int]] {
  def attr: Map[VertexId, Int] = distances
  def toVertex: GraphVertex[Map[VertexId, Int]] = GraphVertex(id, distances)

  /** Returns the landmark distances as an unmodifiable Java {@code Map}. */
  def distancesAsJava: util.Map[java.lang.Long, java.lang.Integer] = {
    val map = new util.HashMap[java.lang.Long, java.lang.Integer](distances.size)
    distances.foreach { case (k, v) =>
      val _ = map.put(java.lang.Long.valueOf(k), java.lang.Integer.valueOf(v))
    }
    util.Collections.unmodifiableMap(map)
  }
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

  /**
   * Runs shortest paths and returns a persisted and materialized dataset of
   * [[ShortestPathsResult]]. Callers should call `.unpersist()` when finished.
   */
  def run(): Dataset[ShortestPathsResult] = {
    val lmarksChecked = check(lmarks, "landmarks")
    val sp = graph.toGraphFrame.shortestPaths
      .landmarks(lmarksChecked)
      .setAlgorithm(algorithm)
      .setCheckpointInterval(checkpointInterval)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setIsDirected(isDirected)

    import AlgorithmCols.*
    val rawDF = sp.run()
    val typed = rawDF
      .select(col(GraphFrame.ID).alias(ID_COL), col(DISTANCES_COL))
      .as[ShortestPathsResult]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
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

  /**
   * Runs triangle counting and returns a persisted and materialized dataset of
   * [[TriangleCountResult]]. Callers should call `.unpersist()` when finished.
   */
  def run(): Dataset[TriangleCountResult] = {
    val tc = graph.toGraphFrame.triangleCount
      .setAlgorithm(algorithm)
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setLgNomEntries(lgNomEntries)

    import AlgorithmCols.*
    val rawDF = tc.run()
    val typed = rawDF
      .select(col(GraphFrame.ID).alias(ID_COL), col(COUNT_COL).cast("long").alias(COUNT_COL))
      .as[TriangleCountResult]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
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

  /**
   * Runs label propagation and returns a persisted and materialized dataset of
   * [[LabelPropagationResult]]. Callers should call `.unpersist()` when finished.
   */
  def run(): Dataset[LabelPropagationResult] = {
    val maxIterChecked = check(maxIter, "maxIter")
    val lp = graph.toGraphFrame.labelPropagation
      .maxIter(maxIterChecked)
      .setAlgorithm(algorithm)
      .setCheckpointInterval(checkpointInterval)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .setIntermediateStorageLevel(intermediateStorageLevel)

    import AlgorithmCols.*
    val rawDF = lp.run()
    // GraphFrames LPA can emit null label for vertices with zero incoming edges
    val typed = rawDF
      .select(
        col(GraphFrame.ID).alias(ID_COL),
        coalesce(col(LABEL_COL), col(GraphFrame.ID)).cast("long").alias(LABEL_COL))
      .as[LabelPropagationResult]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
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
   * Runs strongly connected components and returns a persisted and materialized dataset of
   * [[ConnectedComponentResult]]. Callers should call `.unpersist()` when finished.
   */
  def run(): Dataset[ConnectedComponentResult] = {
    import AlgorithmCols.*
    val maxIterChecked = check(maxIter, "maxIter")
    val rawDF = graph.toGraphFrame.stronglyConnectedComponents
      .maxIter(maxIterChecked)
      .run()
    val typed = rawDF
      .select(col(GraphFrame.ID).alias(ID_COL), col(COMPONENT_COL).alias(CONNECTED_COMPONENT_COL))
      .as[ConnectedComponentResult]
    val res =
      DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, StorageLevel.MEMORY_AND_DISK)
    resultIsPersistent()
    res
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

  /**
   * Runs K-Core decomposition and returns a persisted and materialized dataset of
   * [[KCoreResult]]. Callers should call `.unpersist()` when finished.
   */
  def run(): Dataset[KCoreResult] = {
    import AlgorithmCols.*
    val rawDF = graph.toGraphFrame.kCore
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setCheckpointInterval(checkpointInterval)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .run()
    val typed = rawDF
      .select(col(GraphFrame.ID).alias(ID_COL), col(KCore.kCoreColumnName).alias(KCORE_COL))
      .as[KCoreResult]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
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
   * Runs Maximal Independent Set and returns a persisted and materialized
   * `Dataset[GraphVertex[VD]]` of vertices in the independent set. Callers should call
   * `.unpersist()` when finished.
   */
  def run(seed: Long): Dataset[GraphVertex[VD]] = {
    import AlgorithmCols.*
    val rawMis = graph.toGraphFrame.maximalIndependentSet
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setCheckpointInterval(checkpointInterval)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .run(seed)
    val misIds = rawMis.select(col(GraphFrame.ID).alias(ID_COL))
    val typed = graph.vertices
      .join(misIds, Seq(ID_COL), "left_semi")
      .as[GraphVertex[VD]]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(rawMis, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
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

  /**
   * Runs cycle detection and returns a persisted and materialized `Dataset[Seq[VertexId]]` of
   * unique cycles. Callers should call `.unpersist()` when finished.
   */
  def run(): Dataset[Seq[VertexId]] = {
    val session = graph.spark
    import session.implicits.*
    val rawDF = graph.toGraphFrame.detectingCycles
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .setCheckpointInterval(checkpointInterval)
      .run()
    val typed = rawDF
      .select(col(DetectingCycles.foundSeqCol))
      .as[Seq[VertexId]]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
  }
}

/**
 * Vertex attribute augmented with Parallel Personalized PageRank scores (aligned with the
 * configured `sourceIds`).
 */
case class PersonalizedPageRankAttr[VD](attr: VD, pageranks: Seq[Double]) extends Serializable

object PersonalizedPageRankAttr {
  implicit def encoder[VD: TypeTag]: Encoder[PersonalizedPageRankAttr[VD]] =
    Encoders.product[PersonalizedPageRankAttr[VD]]
}

/**
 * Parallel Personalized PageRank algorithm for [[org.graphframes.GraphDataset]], wrapping
 * [[ParallelPersonalizedPageRank]].
 */
class ParallelPersonalizedPageRankDataset[VD: TypeTag, ED: TypeTag] private[graphframes] (
    private val graph: GraphDataset[VD, ED])
    extends Arguments
    with WithMaxIter
    with Logging {

  private var resetProb: Double = 0.15
  private var srcIds: Array[VertexId] = Array.empty

  /** Source vertices for Personalized PageRank. */
  def sourceIds(values: Array[VertexId]): this.type = {
    srcIds = values
    this
  }

  /** Source vertices for Personalized PageRank. */
  def sourceIds(values: Seq[VertexId]): this.type = {
    srcIds = values.toArray
    this
  }

  /** Reset probability "alpha" (default: 0.15). */
  def resetProbability(value: Double): this.type = {
    resetProb = value
    this
  }

  /**
   * Runs Parallel Personalized PageRank and returns a persisted and materialized [[GraphDataset]]
   * with [[PersonalizedPageRankAttr]] vertex attributes and [[WeightedEdgeAttr]] edge attributes.
   * Callers should call `.unpersist()` on the returned graph when finished.
   */
  def run(): GraphDataset[PersonalizedPageRankAttr[VD], WeightedEdgeAttr[ED]] = {
    val maxIterChecked = check(maxIter, "maxIter")
    val ppr = graph.toGraphFrame.parallelPersonalizedPageRank
      .maxIter(maxIterChecked)
      .resetProbability(resetProb)
      .sourceIds(srcIds.map(id => id: Any))

    import AlgorithmCols.*
    val resGF = ppr.run()
    val vv = resGF.vertices
      .select(
        col(GraphFrame.ID).alias(ID_COL),
        struct(
          col(GraphFrame.ATTR).alias(ATTR_COL),
          vector_to_array(col(PAGERANKS_COL)).alias(PAGERANKS_COL)).alias(ATTR_COL))
      .as[GraphVertex[PersonalizedPageRankAttr[VD]]]
    val ee = resGF.edges
      .select(
        col(GraphFrame.SRC).alias(SRC_ID_COL),
        col(GraphFrame.DST).alias(DST_ID_COL),
        struct(col(GraphFrame.ATTR).alias(ATTR_COL), col(WEIGHT_COL).alias(WEIGHT_COL))
          .alias(ATTR_COL))
      .as[GraphEdge[WeightedEdgeAttr[ED]]]
    val res =
      DatasetCacheUtils.materializeGraphAndReleaseRaw(resGF, vv, ee, StorageLevel.MEMORY_AND_DISK)
    resultIsPersistent()
    res
  }
}

/**
 * Structure-Aware Label Propagation algorithm for [[org.graphframes.GraphDataset]], wrapping
 * [[StructureAwareLabelPropagation]].
 *
 * Note: requires Apache Spark 4.1 or newer.
 */
class StructureAwareLabelPropagationDataset[VD, ED] private[graphframes] (
    private val graph: GraphDataset[VD, ED])
    extends Arguments
    with WithCheckpointInterval
    with WithMaxIter
    with WithLocalCheckpoints
    with WithIntermediateStorageLevel
    with WithDirection
    with WithLgNomEntries
    with Logging {

  private var structuralSimilarityMultiplier: Double = 0.5
  private var ignoreDirectLinks: Boolean = false

  /** Sets whether direct-link baseline vote mass is ignored (default: `false`). */
  def setIgnoreDirectLinks(value: Boolean): this.type = {
    ignoreDirectLinks = value
    this
  }

  /** Sets the multiplier for the neighborhood-overlap signal (default: `0.5`). */
  def setStructuralSimilarityMultiplier(value: Double): this.type = {
    require(value >= 0.0, "structuralSimilarityMultiplier must be >= 0")
    structuralSimilarityMultiplier = value
    this
  }

  private def configure(gf: GraphFrame): StructureAwareLabelPropagation = {
    val maxIterChecked = check(maxIter, "maxIter")
    gf.structureAwareLabelPropagation
      .maxIter(maxIterChecked)
      .setCheckpointInterval(checkpointInterval)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setIsDirected(isDirected)
      .setLgNomEntries(lgNomEntries)
      .setIgnoreDirectLinks(ignoreDirectLinks)
      .setStructuralSimilarityMultiplier(structuralSimilarityMultiplier)
  }

  /**
   * Runs structure-aware label propagation using each vertex's own ID as its initial label,
   * returning a persisted and materialized dataset of [[LabelPropagationResult]].
   */
  def run(): Dataset[LabelPropagationResult] = {
    import AlgorithmCols.*
    val rawDF = configure(graph.toGraphFrame).run()
    val typed = rawDF
      .select(col(GraphFrame.ID).alias(ID_COL), col(LABEL_COL).cast("long").alias(LABEL_COL))
      .as[LabelPropagationResult]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
  }

  /**
   * Runs attribute-guided structure-aware label propagation initialized from a typed function
   * `extractLabel` applied to each vertex, returning a persisted and materialized dataset.
   */
  def runWithInitialLabel[L: TypeTag](
      extractLabel: GraphVertex[VD] => L): Dataset[GraphVertex[L]] = {
    import AlgorithmCols.*
    val labeledV = graph.vertices
      .map(v => GraphVertex(v.id, extractLabel(v)))
      .toDF(GraphFrame.ID, INIT_LABEL_COL)
    val eDF = graph.edges
      .select(col(SRC_ID_COL).alias(GraphFrame.SRC), col(DST_ID_COL).alias(GraphFrame.DST))
    val preparedGF = GraphFrame(labeledV, eDF)
    val rawDF = configure(preparedGF)
      .setInitialLabelCol(INIT_LABEL_COL)
      .run()
    val typed = rawDF
      .select(col(GraphFrame.ID).alias(ID_COL), col(LABEL_COL).alias(ATTR_COL))
      .as[GraphVertex[L]]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
  }
}

/**
 * Result row for [[HyperANFDataset]], containing compact HyperLogLog sketches for hops
 * `0..nHops`.
 *
 * `hopSketches(k)` stores the compact HyperLogLog sketch bytes for vertices reachable in exactly
 * `k` directed hops from `id`.
 */
case class HyperANFResult(id: VertexId, hopSketches: Seq[Array[Byte]]) extends Serializable {
  def numHops: Int = math.max(0, hopSketches.size - 1)
}

object HyperANFResult {
  implicit val encoder: Encoder[HyperANFResult] =
    Encoders.product[HyperANFResult]
}

/**
 * HyperANF approximate neighbourhood function algorithm for [[org.graphframes.GraphDataset]],
 * wrapping [[HyperANF]].
 */
class HyperANFDataset[VD, ED] private[graphframes] (private val graph: GraphDataset[VD, ED])
    extends Serializable
    with Logging
    with WithCheckpointInterval
    with WithIntermediateStorageLevel
    with WithLocalCheckpoints {

  private var nHops: Int = 3
  private var lgNomEntries: Int = 12
  private var edgePredicate: Option[GraphEdge[ED] => Boolean] = None

  /** Sets the log2 of nominal entries used by HLL sketch aggregations (between 4 and 21). */
  def setLgNomEntries(value: Int): this.type = {
    require((value >= 4) && (value <= 21), "lgNomEntries must be between 4 and 21")
    lgNomEntries = value
    this
  }

  /** Sets the maximum hop distance to compute (must be > 0, default: 3). */
  def setNHops(value: Int): this.type = {
    require(value > 0, "n-hops cannot be negative or zero")
    nHops = value
    this
  }

  /** Restricts the HyperANF expansion to edges matching the given typed predicate. */
  def filterEdges(predicate: GraphEdge[ED] => Boolean): this.type = {
    edgePredicate = Some(predicate)
    this
  }

  /**
   * Runs HyperANF and returns a persisted and materialized dataset of [[HyperANFResult]]. Callers
   * should call `.unpersist()` when finished.
   */
  def run(): Dataset[HyperANFResult] = {
    val targetGraph = edgePredicate.map(graph.filterEdges).getOrElse(graph)
    val resDF = targetGraph.toGraphFrame.hyperANF
      .setNHops(nHops)
      .setLgNomEntries(lgNomEntries)
      .setCheckpointInterval(checkpointInterval)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .run()

    import AlgorithmCols.*
    val hopCols = (0 to nHops).map(h => col(s"hop_$h"))
    val typed = resDF
      .select(col(GraphFrame.ID).alias(ID_COL), array(hopCols: _*).alias(HOP_SKETCHES_COL))
      .as[HyperANFResult]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(resDF, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
  }
}

/**
 * Result row for [[RandomWalkEmbeddingsDataset]], containing a vertex ID and embedding vector.
 */
case class NodeEmbeddingResult(id: VertexId, embedding: Seq[Double])
    extends GraphVertexBase[Seq[Double]] {
  def attr: Seq[Double] = embedding
  def toVertex: GraphVertex[Seq[Double]] = GraphVertex(id, embedding)
}

object NodeEmbeddingResult {
  implicit val encoder: Encoder[NodeEmbeddingResult] =
    Encoders.product[NodeEmbeddingResult]
}

/**
 * Random-walk-based node embeddings for [[org.graphframes.GraphDataset]], wrapping
 * [[org.graphframes.embeddings.RandomWalkEmbeddings]].
 */
class RandomWalkEmbeddingsDataset[VD, ED] private[graphframes] (
    private val graph: GraphDataset[VD, ED])
    extends Serializable
    with Logging
    with WithIntermediateStorageLevel {

  private var sequenceModel: Option[Either[Word2Vec, Hash2Vec]] = None
  private var randomWalks: Option[RandomWalkBase] = None
  private var aggregateNeighbors: Boolean = true
  private var useEdgeDirections: Boolean = false
  private var maxNbrs: Int = 50
  private var seed: Long = 42L
  private var cachedRwPath: Option[String] = None
  private var cleanUpAfterRun: Boolean = false

  /** Sets the sequence model (`Left(Word2Vec)` or `Right(Hash2Vec)`). */
  def setSequenceModel(value: Either[Word2Vec, Hash2Vec]): this.type = {
    require(value != null, "sequenceModel must not be null.")
    sequenceModel = Some(value)
    this
  }

  /** Sets the random walk generator. */
  def setRandomWalks(value: RandomWalkBase): this.type = {
    require(value != null, "randomWalks must not be null.")
    randomWalks = Some(value)
    this
  }

  /** Sets the random seed (default: 42L). */
  def setSeed(value: Long): this.type = {
    seed = value
    this
  }

  /** Sets whether to use edge directions (default: `false`). */
  def setUseEdgeDirections(value: Boolean): this.type = {
    useEdgeDirections = value
    this
  }

  /** Sets whether to aggregate neighbor embeddings via min-hash sampling (default: `true`). */
  def setAggregateNeighbors(value: Boolean): this.type = {
    aggregateNeighbors = value
    this
  }

  /** Sets the maximum number of neighbors sampled for aggregation (default: 50). */
  def setMaxNbrs(value: Int): this.type = {
    maxNbrs = value
    this
  }

  /** Sets a path to pre-computed cached random walks in Parquet format. */
  def useCachedRandomWalks(path: String): this.type = {
    cachedRwPath = Some(path)
    this
  }

  /** Sets whether to clean up temporary walk files after running (default: `false`). */
  def setCleanUpAfterRun(value: Boolean): this.type = {
    cleanUpAfterRun = value
    this
  }

  /**
   * Runs random-walk node embeddings and returns a persisted and materialized dataset of
   * [[NodeEmbeddingResult]]. Callers should call `.unpersist()` when finished.
   */
  def run(): Dataset[NodeEmbeddingResult] = {
    require(sequenceModel.isDefined, "sequenceModel must be set before calling run().")
    require(randomWalks.isDefined, "randomWalks must be set before calling run().")

    val rw = graph.toGraphFrame.randomWalksBasedEmbedding
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setSeed(seed)
      .setUseEdgeDirections(useEdgeDirections)
      .setAggregateNeighbors(aggregateNeighbors)
      .setMaxNbrs(maxNbrs)
      .setCleanUpAfterRun(cleanUpAfterRun)
    sequenceModel.foreach(rw.setSequenceModel)
    randomWalks.foreach(rw.setRandomWalks)
    cachedRwPath.foreach(rw.useCachedRandomWalks)

    import AlgorithmCols.*
    val rawDF = rw.run()
    val typed = rawDF
      .select(
        col(GraphFrame.ID).cast("long").alias(ID_COL),
        vector_to_array(col(RandomWalkEmbeddings.embeddingColName)).alias(EMBEDDING_COL))
      .as[NodeEmbeddingResult]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
  }
}

/**
 * Result row for [[SVDPlusPlusDataset]], containing learned explicit factors (`column1`),
 * implicit factors (`column2`), bias (`column3`), and normalization term (`column4`).
 */
case class SVDPlusPlusResult(
    id: VertexId,
    column1: Seq[Double],
    column2: Seq[Double],
    column3: Double,
    column4: Double)
    extends Serializable {

  /** Returns `column1` as an unmodifiable Java {@code List<Double>}. */
  def column1AsJava: util.List[java.lang.Double] = toJavaDoubleList(column1)

  /** Returns `column2` as an unmodifiable Java {@code List<Double>}. */
  def column2AsJava: util.List[java.lang.Double] = toJavaDoubleList(column2)

  private def toJavaDoubleList(values: Seq[Double]): util.List[java.lang.Double] = {
    val list = new util.ArrayList[java.lang.Double](values.size)
    values.foreach { d =>
      val _ = list.add(java.lang.Double.valueOf(d))
    }
    util.Collections.unmodifiableList(list)
  }
}

object SVDPlusPlusResult {
  implicit val encoder: Encoder[SVDPlusPlusResult] =
    Encoders.product[SVDPlusPlusResult]
}

/**
 * SVD++ collaborative filtering algorithm for [[org.graphframes.GraphDataset]], wrapping
 * [[SVDPlusPlus]].
 *
 * Accepts a typed edge weight/rating extractor `ED => Double` (or uses an implicit `Numeric[ED]`
 * when `ED` is already numeric) instead of requiring a `"weight"` column name.
 */
class SVDPlusPlusDataset[VD, ED] private[graphframes] (private val graph: GraphDataset[VD, ED])
    extends Arguments
    with WithMaxIter
    with Logging {

  private var _rank: Int = 10
  private var _minVal: Double = 0.0
  private var _maxVal: Double = 5.0
  private var _gamma1: Double = 0.007
  private var _gamma2: Double = 0.007
  private var _gamma6: Double = 0.005
  private var _gamma7: Double = 0.015
  private var _loss: Option[Double] = None

  def rank(value: Int): this.type = {
    _rank = value
    this
  }

  def minValue(value: Double): this.type = {
    _minVal = value
    this
  }

  def maxValue(value: Double): this.type = {
    _maxVal = value
    this
  }

  def gamma1(value: Double): this.type = {
    _gamma1 = value
    this
  }

  def gamma2(value: Double): this.type = {
    _gamma2 = value
    this
  }

  def gamma6(value: Double): this.type = {
    _gamma6 = value
    this
  }

  def gamma7(value: Double): this.type = {
    _gamma7 = value
    this
  }

  /**
   * Runs SVD++ using `weightExtractor` to extract a rating from each edge attribute `ED`,
   * returning `(vertexEmbeddings, trainingLoss)` where `vertexEmbeddings` is persisted and
   * materialized. Callers should call `.unpersist()` on the returned dataset when finished.
   */
  def run(weightExtractor: ED => Double): (Dataset[SVDPlusPlusResult], Double) = {
    val session = graph.spark
    import session.implicits.*

    val vDF = graph.vertices.toDF(GraphFrame.ID, GraphFrame.ATTR)
    val eDF = graph.edges
      .map(e => (e.srcId, e.dstId, weightExtractor(e.attr)))
      .toDF(GraphFrame.SRC, GraphFrame.DST, SVDPlusPlus.COLUMN_WEIGHT)

    val svd = GraphFrame(vDF, eDF).svdPlusPlus
      .rank(_rank)
      .minValue(_minVal)
      .maxValue(_maxVal)
      .gamma1(_gamma1)
      .gamma2(_gamma2)
      .gamma6(_gamma6)
      .gamma7(_gamma7)
    maxIter.foreach(svd.maxIter)

    val resDF = svd.run()
    val lossVal = svd.loss
    _loss = Some(lossVal)

    import AlgorithmCols.*
    val typed = resDF
      .select(
        col(GraphFrame.ID).alias(ID_COL),
        col(SVDPlusPlus.COLUMN1).alias(COLUMN1_COL),
        col(SVDPlusPlus.COLUMN2).alias(COLUMN2_COL),
        col(SVDPlusPlus.COLUMN3).alias(COLUMN3_COL),
        col(SVDPlusPlus.COLUMN4).alias(COLUMN4_COL))
      .as[SVDPlusPlusResult]

    val persisted =
      DatasetCacheUtils.materializeAndReleaseRaw(resDF, typed, StorageLevel.MEMORY_AND_DISK)
    resultIsPersistent()
    (persisted, lossVal)
  }

  /**
   * Convenience overload for graphs whose edge attribute `ED` is already `Numeric`.
   */
  def run()(implicit num: Numeric[ED]): (Dataset[SVDPlusPlusResult], Double) = {
    run((ed: ED) => num.toDouble(ed))
  }

  /** Returns the training loss from the most recent `run` call. */
  def loss: Double = {
    _loss.getOrElse(throw new IllegalStateException("The algorithm has not been run yet"))
  }
}

/** Result row for Power Iteration Clustering on [[org.graphframes.GraphDataset]]. */
case class ClusterAssignment(id: VertexId, cluster: Int) extends GraphVertexBase[Int] {
  def attr: Int = cluster
  def toVertex: GraphVertex[Int] = GraphVertex(id, cluster)
}

object ClusterAssignment {
  implicit val encoder: Encoder[ClusterAssignment] =
    Encoders.product[ClusterAssignment]
}

/**
 * Result row for typed per-edge-type degree metrics (`typeOutDegree`, `typeInDegree`,
 * `typeDegree`).
 */
case class TypedDegreeResult[K](id: VertexId, degrees: Map[K, Int])
    extends GraphVertexBase[Map[K, Int]] {
  def attr: Map[K, Int] = degrees
  def toVertex: GraphVertex[Map[K, Int]] = GraphVertex(id, degrees)
}

object TypedDegreeResult {
  implicit def encoder[K: TypeTag]: Encoder[TypedDegreeResult[K]] =
    Encoders.product[TypedDegreeResult[K]]
}

private[graphframes] case class TypedDegreePair[K](id: VertexId, k: K) extends Serializable

private[graphframes] object TypedDegreePair {
  implicit def encoder[K: TypeTag]: Encoder[TypedDegreePair[K]] =
    Encoders.product[TypedDegreePair[K]]
}

private[graphframes] case class TypedDegreeOptPair[K](id: VertexId, k: Option[K])
    extends Serializable

private[graphframes] object TypedDegreeOptPair {
  implicit def encoder[K: TypeTag]: Encoder[TypedDegreeOptPair[K]] =
    Encoders.product[TypedDegreeOptPair[K]]
}

private[graphframes] object TypedDegreeMetrics {
  def compute[K: TypeTag](
      pairs: Dataset[TypedDegreePair[K]],
      edgeTypes: Option[Seq[K]]): Dataset[TypedDegreeResult[K]] = {
    import AlgorithmCols.*
    edgeTypes match {
      case Some(allowed) =>
        val allowedList = allowed.filter(k => Option(k).isDefined).distinct.toVector
        val allowedSet = allowedList.toSet
        pairs
          .map { p =>
            val filteredKey = Option(p.k).filter(allowedSet.contains)
            TypedDegreeOptPair(p.id, filteredKey)
          }
          .toDF(ID_COL, KEY_COL)
          .groupBy(col(ID_COL), col(KEY_COL))
          .agg(count("*").cast("int").alias(CNT_COL))
          .groupBy(col(ID_COL))
          .agg(map_from_entries(
            collect_list(when(col(KEY_COL).isNotNull, struct(col(KEY_COL), col(CNT_COL)))))
            .alias(DEGREES_COL))
          .as[TypedDegreeResult[K]]
          .map { r =>
            val existing = Option(r.degrees).getOrElse(Map.empty[K, Int])
            TypedDegreeResult(r.id, allowedList.map(k => k -> existing.getOrElse(k, 0)).toMap)
          }
      case None =>
        pairs
          .toDF(ID_COL, KEY_COL)
          .filter(col(KEY_COL).isNotNull)
          .groupBy(col(ID_COL), col(KEY_COL))
          .agg(count("*").cast("int").alias(CNT_COL))
          .groupBy(col(ID_COL))
          .agg(map_from_entries(collect_list(struct(col(KEY_COL), col(CNT_COL))))
            .alias(DEGREES_COL))
          .as[TypedDegreeResult[K]]
    }
  }
}
