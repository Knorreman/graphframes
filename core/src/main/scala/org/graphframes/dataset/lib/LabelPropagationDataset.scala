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
import org.apache.spark.sql.functions.coalesce
import org.apache.spark.sql.functions.col
import org.graphframes.GraphFrame
import org.graphframes.Logging
import org.graphframes.WithAlgorithmChoice
import org.graphframes.WithCheckpointInterval
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithLocalCheckpoints
import org.graphframes.WithMaxIter
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphVertex
import org.graphframes.dataset.GraphVertexBase
import org.graphframes.dataset.VertexId

/** Result row for [[LabelPropagationDataset]]. */
case class LabelPropagationResult(id: VertexId, label: Long) extends GraphVertexBase[Long] {
  def attr: Long = label
  def toVertex: GraphVertex[Long] = GraphVertex(id, label)
}

object LabelPropagationResult {
  implicit val encoder: Encoder[LabelPropagationResult] =
    Encoders.product[LabelPropagationResult]
}

private object LabelPropagationDatasetCols {
  val ID_COL = "id"
  val LABEL_COL = "label"
}

/**
 * Label Propagation algorithm for [[org.graphframes.dataset.GraphDataset]], wrapping
 * [[org.graphframes.lib.LabelPropagation]].
 */
class LabelPropagationDataset[VD, ED] private[dataset] (private val graph: GraphDataset[VD, ED])
    extends DatasetArguments
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

    import LabelPropagationDatasetCols.*
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
