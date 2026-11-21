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
import org.graphframes.Logging
import org.graphframes.WithCheckpointInterval
import org.graphframes.WithDirection
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithLgNomEntries
import org.graphframes.WithLocalCheckpoints
import org.graphframes.WithMaxIter
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphVertex
import org.graphframes.lib.StructureAwareLabelPropagation

import scala.reflect.runtime.universe.TypeTag

private object StructureAwareLabelPropagationDatasetCols {
  val ID_COL = "id"
  val ATTR_COL = "attr"
  val SRC_ID_COL = "srcId"
  val DST_ID_COL = "dstId"
  val LABEL_COL = "label"
  val INIT_LABEL_COL = "_gf_init_label"
}

/**
 * Structure-Aware Label Propagation algorithm for [[org.graphframes.dataset.GraphDataset]],
 * wrapping [[org.graphframes.lib.StructureAwareLabelPropagation]].
 *
 * Note: requires Apache Spark 4.1 or newer.
 */
class StructureAwareLabelPropagationDataset[VD, ED] private[dataset] (
    private val graph: GraphDataset[VD, ED])
    extends DatasetArguments
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
    import StructureAwareLabelPropagationDatasetCols.*
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
    import StructureAwareLabelPropagationDatasetCols.*
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
