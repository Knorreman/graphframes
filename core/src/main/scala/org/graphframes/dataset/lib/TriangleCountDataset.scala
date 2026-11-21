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
import org.graphframes.GraphFrame
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithLgNomEntries
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphVertex
import org.graphframes.dataset.GraphVertexBase
import org.graphframes.dataset.VertexId

/** Result row for [[TriangleCountDataset]]. */
case class TriangleCountResult(id: VertexId, count: Long) extends GraphVertexBase[Long] {
  def attr: Long = count
  def toVertex: GraphVertex[Long] = GraphVertex(id, count)
}

object TriangleCountResult {
  implicit val encoder: Encoder[TriangleCountResult] =
    Encoders.product[TriangleCountResult]
}

private object TriangleCountDatasetCols {
  val ID_COL = "id"
  val COUNT_COL = "count"
}

/**
 * Triangle Count algorithm for [[org.graphframes.dataset.GraphDataset]], wrapping
 * [[org.graphframes.lib.TriangleCount]].
 */
class TriangleCountDataset[VD, ED] private[dataset] (private val graph: GraphDataset[VD, ED])
    extends DatasetArguments
    with Serializable
    with WithIntermediateStorageLevel
    with WithLgNomEntries {

  private var algorithm: String = "exact"
  private val supportedAlgorithms: Set[String] = Set("exact", "approx")

  /** Sets the triangle counting algorithm: `"exact"` (default) or `"approx"`. */
  def setAlgorithm(value: String): this.type = {
    require(
      value != null && supportedAlgorithms.contains(value),
      s"supported algorithms: ${supportedAlgorithms.mkString(", ")}")
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

    import TriangleCountDatasetCols.*
    val rawDF = tc.run()
    val typed = rawDF
      .select(col(GraphFrame.ID).alias(ID_COL), col(COUNT_COL).cast("long").alias(COUNT_COL))
      .as[TriangleCountResult]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
  }
}
