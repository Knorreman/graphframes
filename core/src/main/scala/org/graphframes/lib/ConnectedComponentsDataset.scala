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

package org.graphframes.lib

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.col
import org.graphframes.GraphDataset
import org.graphframes.GraphFrame
import org.graphframes.Logging
import org.graphframes.WithBroadcastThreshold
import org.graphframes.WithCheckpointInterval
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithLocalCheckpoints
import org.graphframes.WithMaxIter

/**
 * Connected Components algorithm for [[org.graphframes.GraphDataset]].
 *
 * Wraps [[ConnectedComponents]] on the underlying [[GraphFrame]] representation and returns a
 * typed `Dataset[ConnectedComponentResult]` assigning each vertex to its connected component ID.
 */
class ConnectedComponentsDataset[VD, ED] private[graphframes] (
    private val graph: GraphDataset[VD, ED])
    extends Arguments
    with Logging
    with WithBroadcastThreshold
    with WithCheckpointInterval
    with WithIntermediateStorageLevel
    with WithLocalCheckpoints
    with WithMaxIter {

  private var algorithm: String = ConnectedComponents.ALGO_TWO_PHASE

  /**
   * Sets the algorithm to use for computing connected components. Supported values match
   * [[ConnectedComponents.setAlgorithm]] (`"two_phase"`, `"randomized_contraction"`, `"graphx"`).
   */
  def setAlgorithm(value: String): this.type = {
    algorithm = value
    this
  }

  /** Gets the algorithm used for computing connected components. */
  def getAlgorithm: String = algorithm

  /**
   * Runs the connected components algorithm by delegating to [[GraphFrame.connectedComponents]]
   * and returns a typed dataset of [[ConnectedComponentResult]].
   */
  def run(): Dataset[ConnectedComponentResult] = {
    val cc = graph.toGraphFrame.connectedComponents
      .setAlgorithm(algorithm)
      .setBroadcastThreshold(broadcastThreshold)
      .setCheckpointInterval(checkpointInterval)
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setUseLocalCheckpoints(useLocalCheckpoints)
    maxIter.foreach(cc.maxIter)

    cc.run()
      .select(
        col(GraphFrame.ID).alias("id"),
        col(ConnectedComponents.COMPONENT).alias("connectedComponent"))
      .as[ConnectedComponentResult]
  }
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
