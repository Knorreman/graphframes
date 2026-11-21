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
import org.graphframes.WithDirection
import org.graphframes.WithIntermediateStorageLevel
import org.graphframes.WithLocalCheckpoints
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphEdge
import org.graphframes.dataset.GraphVertex
import org.graphframes.dataset.VertexId

import java.util

/**
 * A simple path represented by ordered vertex IDs and its edge count.
 */
case class SimplePath(path: Seq[VertexId], length: Int) extends Serializable {

  /** The starting vertex ID. */
  def from: VertexId = path.head

  /** The destination vertex ID. */
  def to: VertexId = path.last

  /** Returns the ordered vertex IDs in this path as an unmodifiable Java {@code List}. */
  def pathAsJava: util.List[java.lang.Long] = {
    val list = new util.ArrayList[java.lang.Long](path.size)
    path.foreach { id =>
      val _ = list.add(java.lang.Long.valueOf(id))
    }
    util.Collections.unmodifiableList(list)
  }
}

object SimplePath {
  implicit val encoder: Encoder[SimplePath] = Encoders.product[SimplePath]
}

private object AllPathsDatasetCols {
  val SRC_ID_COL = "srcId"
  val DST_ID_COL = "dstId"
  val PATH_COL = "path"
  val LEN_COL = "len"
  val LENGTH_COL = "length"
  val ALL_PATHS_FROM_FLAG = "_gf_ap_from"
  val ALL_PATHS_TO_FLAG = "_gf_ap_to"
}

/**
 * Typed AllPaths algorithm for [[org.graphframes.dataset.GraphDataset]], wrapping
 * [[org.graphframes.lib.AllPaths]].
 *
 * Enumerates all simple paths (no repeated vertices) up to `maxPathLength` hops between vertices
 * matching `fromPred` and vertices matching `toPred`.
 */
class AllPathsDataset[VD, ED] private[dataset] (private val graph: GraphDataset[VD, ED])
    extends DatasetArguments
    with Serializable
    with WithDirection
    with WithLocalCheckpoints
    with WithCheckpointInterval
    with WithIntermediateStorageLevel {

  private var maxPathLen: Int = 5
  private var fromPredicate: Option[GraphVertex[VD] => Boolean] = None
  private var toPredicate: Option[GraphVertex[VD] => Boolean] = None
  private var edgePredicate: Option[GraphEdge[ED] => Boolean] = None

  /** Sets the typed predicate identifying source vertices. */
  def fromPred(predicate: GraphVertex[VD] => Boolean): this.type = {
    fromPredicate = Some(predicate)
    this
  }

  /** Sets the typed predicate identifying destination vertices. */
  def toPred(predicate: GraphVertex[VD] => Boolean): this.type = {
    toPredicate = Some(predicate)
    this
  }

  /** Sets an optional typed predicate restricting which edges may be traversed. */
  def edgeFilter(predicate: GraphEdge[ED] => Boolean): this.type = {
    edgePredicate = Some(predicate)
    this
  }

  /** Sets the maximum path length (number of edges, > 0, default: 5). */
  def maxPathLength(value: Int): this.type = {
    require(value > 0, s"AllPaths maxPathLength must be > 0, but was set to $value")
    maxPathLen = value
    this
  }

  /** Executes AllPaths and returns a typed `Dataset[SimplePath]`. */
  def run(): Dataset[SimplePath] = {
    val fromFn =
      fromPredicate.getOrElse(throw new IllegalArgumentException("fromPred is required."))
    val toFn = toPredicate.getOrElse(throw new IllegalArgumentException("toPred is required."))

    import AllPathsDatasetCols.*
    val spark = graph.spark
    import spark.implicits.*

    val vDF = graph.vertices
      .map(v => (v.id, fromFn(v), toFn(v)))
      .toDF(GraphFrame.ID, ALL_PATHS_FROM_FLAG, ALL_PATHS_TO_FLAG)

    val filteredEdges = edgePredicate.map(graph.edges.filter).getOrElse(graph.edges)
    val eDF = filteredEdges
      .select(col(SRC_ID_COL).alias(GraphFrame.SRC), col(DST_ID_COL).alias(GraphFrame.DST))
      .distinct()

    GraphFrame(vDF, eDF).allPaths
      .fromExpr(col(ALL_PATHS_FROM_FLAG))
      .toExpr(col(ALL_PATHS_TO_FLAG))
      .maxPathLength(maxPathLen)
      .setIsDirected(isDirected)
      .setUseLocalCheckpoints(useLocalCheckpoints)
      .setCheckpointInterval(checkpointInterval)
      .setIntermediateStorageLevel(intermediateStorageLevel)
      .run()
      .select(col(PATH_COL), col(LEN_COL).cast("int").alias(LENGTH_COL))
      .as[SimplePath]
  }
}
