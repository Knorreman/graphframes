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
import org.apache.spark.sql.functions.size
import org.graphframes.Logging
import org.graphframes.WithCheckpointInterval
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithLocalCheckpoints
import org.graphframes.dataset.GraphDataset
import org.graphframes.lib.DetectingCycles

private object DetectingCyclesDatasetCols {
  val PATH_COL = "path"
  val LENGTH_COL = "length"
}

/**
 * Cycle detection algorithm for [[org.graphframes.dataset.GraphDataset]], wrapping
 * [[org.graphframes.lib.DetectingCycles]].
 */
class DetectingCyclesDataset[VD, ED] private[dataset] (private val graph: GraphDataset[VD, ED])
    extends DatasetArguments
    with Serializable
    with Logging
    with WithIntermediateStorageLevel
    with WithLocalCheckpoints
    with WithCheckpointInterval {

  /**
   * Runs cycle detection and returns a persisted and materialized `Dataset[SimplePath]` of unique
   * cycles. Callers should call `.unpersist()` when finished.
   */
  def run(): Dataset[SimplePath] = {
    import DetectingCyclesDatasetCols.*
    val rawDF = graph.toGraphFrame.detectingCycles
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .setCheckpointInterval(checkpointInterval)
      .run()
    val typed = rawDF
      .select(
        col(DetectingCycles.foundSeqCol).alias(PATH_COL),
        (size(col(DetectingCycles.foundSeqCol)) - 1).alias(LENGTH_COL))
      .as[SimplePath]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
  }
}
