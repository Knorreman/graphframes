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
import org.apache.spark.sql.graphframes.GraphFramesConf
import org.apache.spark.storage.StorageLevel
import org.graphframes.GraphFrame
import org.graphframes.Logging
import org.graphframes.WithBroadcastThreshold
import org.graphframes.WithCheckpointInterval
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithLocalCheckpoints
import org.graphframes.WithMaxIter
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphVertex
import org.graphframes.dataset.GraphVertexBase
import org.graphframes.dataset.VertexId
import org.graphframes.lib.ConnectedComponents

/**
 * Connected Components algorithm for [[org.graphframes.dataset.GraphDataset]].
 *
 * Wraps [[org.graphframes.lib.ConnectedComponents]] on the underlying [[GraphFrame]]
 * representation and returns a persisted and materialized `Dataset[ConnectedComponentResult]`
 * assigning each vertex to its connected component ID. Callers should call `.unpersist()` on the
 * returned dataset when it is no longer needed. Unset configuration parameters fall back to
 * `spark.graphframes.*` SQL configurations via `GraphFramesConf`.
 */
class ConnectedComponentsDataset[VD, ED] private[dataset] (
    private val graph: GraphDataset[VD, ED])
    extends DatasetArguments
    with Logging
    with WithBroadcastThreshold
    with WithCheckpointInterval
    with WithIntermediateStorageLevel
    with WithLocalCheckpoints
    with WithMaxIter {

  private var algorithmOpt: Option[String] = None
  private var broadcastThresholdOpt: Option[Int] = None
  private var checkpointIntervalOpt: Option[Int] = None
  private var intermediateStorageLevelOpt: Option[StorageLevel] = None
  private var useLocalCheckpointsOpt: Option[Boolean] = None

  /**
   * Sets the algorithm to use for computing connected components. Supported values match
   * [[org.graphframes.lib.ConnectedComponents.setAlgorithm]] (`"two_phase"`,
   * `"randomized_contraction"`, `"graphx"`).
   */
  def setAlgorithm(value: String): this.type = {
    require(value != null, "algorithm must not be null.")
    val normalized = value.toLowerCase(java.util.Locale.ROOT)
    normalized match {
      case ConnectedComponents.ALGO_GRAPHX | ConnectedComponents.ALGO_TWO_PHASE |
          ConnectedComponents.ALGO_RANDOMIZED_CONTRACTION =>
        algorithmOpt = Some(normalized)
      case ConnectedComponents.ALGO_GRAPHFRAMES =>
        logWarn(
          s"Algorithm '${ConnectedComponents.ALGO_GRAPHFRAMES}' is deprecated and will be removed " +
            s"in a future release. Using '${ConnectedComponents.ALGO_TWO_PHASE}' instead.")
        algorithmOpt = Some(ConnectedComponents.ALGO_TWO_PHASE)
      case _ =>
        throw new IllegalArgumentException(
          s"Unsupported algorithm: '$value'. " +
            s"Supported values are: ${ConnectedComponents.ALGO_GRAPHX}, " +
            s"${ConnectedComponents.ALGO_TWO_PHASE}, " +
            s"${ConnectedComponents.ALGO_RANDOMIZED_CONTRACTION}, " +
            s"${ConnectedComponents.ALGO_GRAPHFRAMES} (deprecated).")
    }
    this
  }

  /** Gets the algorithm used for computing connected components. */
  def getAlgorithm: String =
    algorithmOpt
      .orElse(GraphFramesConf.getConnectedComponentsAlgorithm)
      .getOrElse(ConnectedComponents.ALGO_TWO_PHASE)

  override def setBroadcastThreshold(value: Int): this.type = {
    broadcastThresholdOpt = Some(value)
    super.setBroadcastThreshold(value)
  }

  override def getBroadcastThreshold: Int =
    broadcastThresholdOpt
      .orElse(GraphFramesConf.getConnectedComponentsBroadcastThreshold)
      .getOrElse(super.getBroadcastThreshold)

  override def setCheckpointInterval(value: Int): this.type = {
    checkpointIntervalOpt = Some(value)
    super.setCheckpointInterval(value)
  }

  override def getCheckpointInterval: Int =
    checkpointIntervalOpt
      .orElse(GraphFramesConf.getConnectedComponentsCheckpointInterval)
      .getOrElse(super.getCheckpointInterval)

  override def setIntermediateStorageLevel(value: StorageLevel): this.type = {
    intermediateStorageLevelOpt = Some(value)
    super.setIntermediateStorageLevel(value)
  }

  override def getIntermediateStorageLevel: StorageLevel =
    intermediateStorageLevelOpt
      .orElse(GraphFramesConf.getConnectedComponentsStorageLevel)
      .getOrElse(super.getIntermediateStorageLevel)

  override def setUseLocalCheckpoints(value: Boolean): this.type = {
    useLocalCheckpointsOpt = Some(value)
    super.setUseLocalCheckpoints(value)
  }

  override def getUseLocalCheckpoints: Boolean =
    useLocalCheckpointsOpt
      .orElse(GraphFramesConf.getUseLocalCheckpoints)
      .getOrElse(super.getUseLocalCheckpoints)

  /**
   * Runs the connected components algorithm by delegating to [[GraphFrame.connectedComponents]]
   * and returns a persisted and materialized dataset of [[ConnectedComponentResult]].
   */
  def run(): Dataset[ConnectedComponentResult] = {
    val cc = graph.toGraphFrame.connectedComponents
    algorithmOpt.foreach(cc.setAlgorithm)
    broadcastThresholdOpt.foreach(cc.setBroadcastThreshold)
    checkpointIntervalOpt.foreach(cc.setCheckpointInterval)
    intermediateStorageLevelOpt.foreach(cc.setIntermediateStorageLevel)
    useLocalCheckpointsOpt.foreach(cc.setUseLocalCheckpoints)
    maxIter.foreach(cc.maxIter)

    val rawDF = cc.run()
    val typed = rawDF
      .select(
        col(GraphFrame.ID).alias(ConnectedComponentsDatasetCols.ID_COL),
        col(ConnectedComponents.COMPONENT)
          .alias(ConnectedComponentsDatasetCols.CONNECTED_COMPONENT_COL))
      .as[ConnectedComponentResult]
    val res =
      DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, cc.getIntermediateStorageLevel)
    resultIsPersistent()
    res
  }
}

private object ConnectedComponentsDatasetCols {
  val ID_COL = "id"
  val CONNECTED_COMPONENT_COL = "connectedComponent"
}

/**
 * Result row for [[ConnectedComponentsDataset]], mapping each vertex ID to its connected
 * component ID.
 */
case class ConnectedComponentResult(id: VertexId, connectedComponent: Long)
    extends GraphVertexBase[Long] {
  def attr: Long = connectedComponent
  def toVertex: GraphVertex[Long] = GraphVertex(id, connectedComponent)
}

object ConnectedComponentResult {
  implicit val encoder: Encoder[ConnectedComponentResult] =
    Encoders.product[ConnectedComponentResult]
}
