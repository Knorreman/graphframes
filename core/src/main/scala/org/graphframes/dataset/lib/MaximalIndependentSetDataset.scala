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
import org.apache.spark.sql.functions.col
import org.graphframes.GraphFrame
import org.graphframes.WithCheckpointInterval
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithLocalCheckpoints
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphVertex

import scala.reflect.runtime.universe.TypeTag

private object MaximalIndependentSetDatasetCols {
  val ID_COL = "id"
}

/**
 * Maximal Independent Set algorithm for [[org.graphframes.dataset.GraphDataset]], wrapping
 * [[org.graphframes.lib.MaximalIndependentSet]].
 */
class MaximalIndependentSetDataset[VD: TypeTag, ED] private[dataset] (
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
    import MaximalIndependentSetDatasetCols.*
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
