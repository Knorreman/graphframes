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
import org.graphframes.WithCheckpointInterval
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithLocalCheckpoints
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphVertex
import org.graphframes.dataset.GraphVertexBase
import org.graphframes.dataset.VertexId
import org.graphframes.lib.KCore

/** Result row for [[KCoreDataset]]. */
case class KCoreResult(id: VertexId, kcore: Int) extends GraphVertexBase[Int] {
  def attr: Int = kcore
  def toVertex: GraphVertex[Int] = GraphVertex(id, kcore)
}

object KCoreResult {
  implicit val encoder: Encoder[KCoreResult] =
    Encoders.product[KCoreResult]
}

private object KCoreDatasetCols {
  val ID_COL = "id"
  val KCORE_COL = "kcore"
}

/**
 * K-Core decomposition algorithm for [[org.graphframes.dataset.GraphDataset]], wrapping
 * [[org.graphframes.lib.KCore]].
 */
class KCoreDataset[VD, ED] private[dataset] (private val graph: GraphDataset[VD, ED])
    extends Serializable
    with WithIntermediateStorageLevel
    with WithCheckpointInterval
    with WithLocalCheckpoints {

  /**
   * Runs K-Core decomposition and returns a persisted and materialized dataset of
   * [[KCoreResult]]. Callers should call `.unpersist()` when finished.
   */
  def run(): Dataset[KCoreResult] = {
    import KCoreDatasetCols.*
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
