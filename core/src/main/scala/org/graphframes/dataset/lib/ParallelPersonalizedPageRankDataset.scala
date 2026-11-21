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

import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.struct
import org.apache.spark.storage.StorageLevel
import org.graphframes.GraphFrame
import org.graphframes.Logging
import org.graphframes.WithMaxIter
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphEdge
import org.graphframes.dataset.GraphVertex
import org.graphframes.dataset.VertexId

import scala.reflect.runtime.universe.TypeTag

/**
 * Vertex attribute augmented with Parallel Personalized PageRank scores (aligned with the
 * configured `sourceIds`).
 */
case class PersonalizedPageRankAttr[VD](attr: VD, pageranks: Seq[Double]) extends Serializable

object PersonalizedPageRankAttr {
  implicit def encoder[VD: TypeTag]: Encoder[PersonalizedPageRankAttr[VD]] =
    Encoders.product[PersonalizedPageRankAttr[VD]]
}

private object ParallelPersonalizedPageRankDatasetCols {
  val ID_COL = "id"
  val ATTR_COL = "attr"
  val SRC_ID_COL = "srcId"
  val DST_ID_COL = "dstId"
  val PAGERANKS_COL = "pageranks"
  val WEIGHT_COL = "weight"
}

/**
 * Parallel Personalized PageRank algorithm for [[org.graphframes.dataset.GraphDataset]], wrapping
 * [[org.graphframes.lib.ParallelPersonalizedPageRank]].
 */
class ParallelPersonalizedPageRankDataset[VD: TypeTag, ED: TypeTag] private[dataset] (
    private val graph: GraphDataset[VD, ED])
    extends DatasetArguments
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
   * Runs Parallel Personalized PageRank and returns a persisted and materialized
   * [[org.graphframes.dataset.GraphDataset]] with [[PersonalizedPageRankAttr]] vertex attributes
   * and [[WeightedEdgeAttr]] edge attributes. Callers should call `.unpersist()` on the returned
   * graph when finished.
   */
  def run(): GraphDataset[PersonalizedPageRankAttr[VD], WeightedEdgeAttr[ED]] = {
    val maxIterChecked = check(maxIter, "maxIter")
    val ppr = graph.toGraphFrame.parallelPersonalizedPageRank
      .maxIter(maxIterChecked)
      .resetProbability(resetProb)
      .sourceIds(srcIds.map(id => id: Any))

    import ParallelPersonalizedPageRankDatasetCols.*
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
