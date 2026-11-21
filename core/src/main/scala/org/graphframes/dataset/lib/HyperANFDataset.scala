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
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.functions.col
import org.graphframes.GraphFrame
import org.graphframes.Logging
import org.graphframes.WithCheckpointInterval
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithLocalCheckpoints
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphEdge
import org.graphframes.dataset.VertexId

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

private object HyperANFDatasetCols {
  val ID_COL = "id"
  val HOP_SKETCHES_COL = "hopSketches"
}

/**
 * HyperANF approximate neighbourhood function algorithm for
 * [[org.graphframes.dataset.GraphDataset]], wrapping [[org.graphframes.lib.HyperANF]].
 */
class HyperANFDataset[VD, ED] private[dataset] (private val graph: GraphDataset[VD, ED])
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

    import HyperANFDatasetCols.*
    val hopCols = (0 to nHops).map(h => col(s"hop_$h"))
    val typed = resDF
      .select(col(GraphFrame.ID).alias(ID_COL), array(hopCols: _*).alias(HOP_SKETCHES_COL))
      .as[HyperANFResult]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(resDF, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
  }
}
