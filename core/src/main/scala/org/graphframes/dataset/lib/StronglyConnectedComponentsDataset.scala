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
import org.apache.spark.storage.StorageLevel
import org.graphframes.GraphFrame
import org.graphframes.Logging
import org.graphframes.WithMaxIter
import org.graphframes.dataset.GraphDataset

private object StronglyConnectedComponentsDatasetCols {
  val ID_COL = "id"
  val COMPONENT_COL = "component"
  val CONNECTED_COMPONENT_COL = "connectedComponent"
}

/**
 * Strongly Connected Components algorithm for [[org.graphframes.dataset.GraphDataset]], wrapping
 * [[org.graphframes.lib.StronglyConnectedComponents]].
 */
class StronglyConnectedComponentsDataset[VD, ED] private[dataset] (
    private val graph: GraphDataset[VD, ED])
    extends DatasetArguments
    with WithMaxIter
    with Logging {

  /**
   * Runs strongly connected components and returns a persisted and materialized dataset of
   * [[ConnectedComponentResult]]. Callers should call `.unpersist()` when finished.
   */
  def run(): Dataset[ConnectedComponentResult] = {
    import StronglyConnectedComponentsDatasetCols.*
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
