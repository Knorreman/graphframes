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
import org.graphframes.Logging
import org.graphframes.WithAlgorithmChoice
import org.graphframes.WithCheckpointInterval
import org.graphframes.WithDirection
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithLocalCheckpoints
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphVertex
import org.graphframes.dataset.GraphVertexBase
import org.graphframes.dataset.VertexId

import java.util

/** Result row for [[ShortestPathsDataset]], mapping each vertex to landmark distances. */
case class ShortestPathsResult(id: VertexId, distances: Map[VertexId, Int])
    extends GraphVertexBase[Map[VertexId, Int]] {
  def attr: Map[VertexId, Int] = distances
  def toVertex: GraphVertex[Map[VertexId, Int]] = GraphVertex(id, distances)

  /** Returns the landmark distances as an unmodifiable Java {@code Map}. */
  def distancesAsJava: util.Map[java.lang.Long, java.lang.Integer] = {
    val map = new util.HashMap[java.lang.Long, java.lang.Integer](distances.size)
    distances.foreach { case (k, v) =>
      val _ = map.put(java.lang.Long.valueOf(k), java.lang.Integer.valueOf(v))
    }
    util.Collections.unmodifiableMap(map)
  }
}

object ShortestPathsResult {
  implicit val encoder: Encoder[ShortestPathsResult] =
    Encoders.product[ShortestPathsResult]
}

private object ShortestPathsDatasetCols {
  val ID_COL = "id"
  val DISTANCES_COL = "distances"
}

/**
 * Shortest Paths algorithm for [[org.graphframes.dataset.GraphDataset]], wrapping
 * [[org.graphframes.lib.ShortestPaths]].
 */
class ShortestPathsDataset[VD, ED] private[dataset] (private val graph: GraphDataset[VD, ED])
    extends DatasetArguments
    with WithAlgorithmChoice
    with WithCheckpointInterval
    with WithLocalCheckpoints
    with WithIntermediateStorageLevel
    with WithDirection
    with Logging {

  private var lmarks: Option[Seq[VertexId]] = None

  /** The list of landmark vertex IDs to compute shortest paths to. */
  def landmarks(value: Seq[VertexId]): this.type = {
    lmarks = Some(value)
    this
  }

  /**
   * Runs shortest paths and returns a persisted and materialized dataset of
   * [[ShortestPathsResult]]. Callers should call `.unpersist()` when finished.
   */
  def run(): Dataset[ShortestPathsResult] = {
    val lmarksChecked = check(lmarks, "landmarks")
    val sp = graph.toGraphFrame.shortestPaths
      .landmarks(lmarksChecked)
      .setAlgorithm(algorithm)
      .setCheckpointInterval(checkpointInterval)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .setIsDirected(isDirected)

    import ShortestPathsDatasetCols.*
    val rawDF = sp.run()
    val typed = rawDF
      .select(col(GraphFrame.ID).alias(ID_COL), col(DISTANCES_COL))
      .as[ShortestPathsResult]
    val res = DatasetCacheUtils.materializeAndReleaseRaw(rawDF, typed, intermediateStorageLevel)
    resultIsPersistent()
    res
  }
}
