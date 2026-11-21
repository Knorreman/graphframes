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

import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.struct
import org.apache.spark.storage.StorageLevel
import org.graphframes.GraphFrame
import org.graphframes.Logging
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphEdge
import org.graphframes.dataset.GraphVertex
import org.graphframes.dataset.VertexId

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

private object PageRankDatasetCols {
  val ID_COL = "id"
  val ATTR_COL = "attr"
  val SRC_ID_COL = "srcId"
  val DST_ID_COL = "dstId"
  val PAGERANK_COL = "pagerank"
  val WEIGHT_COL = "weight"
}

/**
 * PageRank algorithm for [[org.graphframes.dataset.GraphDataset]], wrapping
 * [[org.graphframes.lib.PageRank]].
 */
class PageRankDataset[VD: TypeTag, ED: TypeTag] private[dataset] (
    private val graph: GraphDataset[VD, ED])
    extends DatasetArguments
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
   * Runs PageRank and returns a persisted and materialized
   * [[org.graphframes.dataset.GraphDataset]] whose vertex attributes are [[PageRankAttr]] and
   * edge attributes are [[WeightedEdgeAttr]]. Callers should call `.unpersist()` on the returned
   * graph when it is no longer needed.
   */
  def run(): GraphDataset[PageRankAttr[VD], WeightedEdgeAttr[ED]] = {
    val pr = graph.toGraphFrame.pageRank.resetProbability(resetProb)
    srcIdOpt.foreach(pr.sourceId)
    tolOpt.foreach(pr.tol)
    maxIterOpt.foreach(pr.maxIter)

    import PageRankDatasetCols.*
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
