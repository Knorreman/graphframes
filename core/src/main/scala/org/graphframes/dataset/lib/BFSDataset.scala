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
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.struct
import org.apache.spark.sql.functions.typedLit
import org.graphframes.GraphFrame
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphEdge
import org.graphframes.dataset.GraphVertex
import org.graphframes.dataset.VertexId

import scala.reflect.runtime.universe.TypeTag

/**
 * A typed graph path consisting of ordered vertices and the edges connecting them.
 *
 * For a 0-hop path (where a vertex matches both source and destination predicates), `vertices`
 * has 1 element and `edges` is empty (`length == 0`). For a `k`-hop path, `vertices.size == k +
 * 1` and `edges.size == k`.
 */
case class GraphPath[VD, ED](vertices: Seq[GraphVertex[VD]], edges: Seq[GraphEdge[ED]])
    extends Serializable {

  /** The starting vertex of the path. */
  def from: GraphVertex[VD] = vertices.head

  /** The ending vertex of the path. */
  def to: GraphVertex[VD] = vertices.last

  /** The number of edges in the path. */
  def length: Int = edges.size
}

object GraphPath {
  implicit def encoder[VD: TypeTag, ED: TypeTag]: Encoder[GraphPath[VD, ED]] =
    Encoders.product[GraphPath[VD, ED]]
}

private[lib] case class BFSFlaggedVertex[VD](
    id: VertexId,
    attr: VD,
    _gf_bfs_from: Boolean,
    _gf_bfs_to: Boolean)
    extends Serializable

private[lib] object BFSFlaggedVertex {
  implicit def encoder[VD: TypeTag]: Encoder[BFSFlaggedVertex[VD]] =
    Encoders.product[BFSFlaggedVertex[VD]]
}

private object BFSDatasetCols {
  val ID_COL = "id"
  val ATTR_COL = "attr"
  val SRC_ID_COL = "srcId"
  val DST_ID_COL = "dstId"
  val FROM_COL = "from"
  val TO_COL = "to"
  val VERTICES_COL = "vertices"
  val EDGES_COL = "edges"
  val BFS_FROM_FLAG = "_gf_bfs_from"
  val BFS_TO_FLAG = "_gf_bfs_to"
}

/**
 * Typed Breadth-First Search (BFS) for [[org.graphframes.dataset.GraphDataset]].
 *
 * Finds all globally shortest paths from vertices matching `fromPred` to vertices matching
 * `toPred` up to `maxPathLength` hops, returning a typed `Dataset[GraphPath[VD, ED]]`.
 */
class BFSDataset[VD: TypeTag, ED: TypeTag] private[dataset] (
    private val graph: GraphDataset[VD, ED])
    extends DatasetArguments
    with Serializable {

  private var maxPathLen: Int = 10
  private var fromPredicate: Option[GraphVertex[VD] => Boolean] = None
  private var toPredicate: Option[GraphVertex[VD] => Boolean] = None
  private var edgePredicate: Option[GraphEdge[ED] => Boolean] = None

  /** Sets the typed predicate identifying starting vertices. */
  def fromPred(predicate: GraphVertex[VD] => Boolean): this.type = {
    fromPredicate = Some(predicate)
    this
  }

  /** Sets the typed predicate identifying target vertices. */
  def toPred(predicate: GraphVertex[VD] => Boolean): this.type = {
    toPredicate = Some(predicate)
    this
  }

  /** Sets an optional typed predicate restricting which edges may be traversed. */
  def edgeFilter(predicate: GraphEdge[ED] => Boolean): this.type = {
    edgePredicate = Some(predicate)
    this
  }

  /** Sets the maximum path length (number of edges, >= 0, default: 10). */
  def maxPathLength(value: Int): this.type = {
    require(value >= 0, s"BFS maxPathLength must be >= 0, but was set to $value")
    maxPathLen = value
    this
  }

  /** Executes BFS and returns a typed `Dataset[GraphPath[VD, ED]]`. */
  def run(): Dataset[GraphPath[VD, ED]] = {
    val fromFn =
      fromPredicate.getOrElse(throw new IllegalArgumentException("fromPred is required."))
    val toFn = toPredicate.getOrElse(throw new IllegalArgumentException("toPred is required."))

    import BFSDatasetCols.*
    val spark = graph.spark

    val vDF = graph.vertices
      .map(v => BFSFlaggedVertex(v.id, v.attr, fromFn(v), toFn(v)))
      .toDF(GraphFrame.ID, GraphFrame.ATTR, BFS_FROM_FLAG, BFS_TO_FLAG)

    val filteredEdges = edgePredicate.map(graph.edges.filter).getOrElse(graph.edges)
    val eDF = filteredEdges.toDF(GraphFrame.SRC, GraphFrame.DST, GraphFrame.ATTR)

    val resDF = GraphFrame(vDF, eDF).bfs
      .fromExpr(col(BFS_FROM_FLAG))
      .toExpr(col(BFS_TO_FLAG))
      .maxPathLength(maxPathLen)
      .run()

    val cols = resDF.columns
    if (!cols.contains(FROM_COL) || !cols.contains(TO_COL)) {
      return spark.emptyDataset[GraphPath[VD, ED]]
    }

    val edgeCols =
      cols.filter(c => c.startsWith("e") && c.length > 1 && c.drop(1).forall(_.isDigit))
    if (edgeCols.isEmpty) {
      // 0-hop paths: from == to
      resDF
        .select(
          array(
            struct(
              col(s"$FROM_COL.$ID_COL").alias(ID_COL),
              col(s"$FROM_COL.$ATTR_COL").alias(ATTR_COL)))
            .alias(VERTICES_COL),
          typedLit(Seq.empty[GraphEdge[ED]]).alias(EDGES_COL))
        .as[GraphPath[VD, ED]]
    } else {
      val vertexCols = cols.filter(c =>
        c == FROM_COL || c == TO_COL ||
          (c.startsWith("v") && c.length > 1 && c.drop(1).forall(_.isDigit)))
      val verticesExpr = array(
        vertexCols
          .map(c =>
            struct(col(s"$c.$ID_COL").alias(ID_COL), col(s"$c.$ATTR_COL").alias(ATTR_COL)))
          .toIndexedSeq: _*)
      val edgesExpr = array(
        edgeCols
          .map(c =>
            struct(
              col(s"$c.${GraphFrame.SRC}").alias(SRC_ID_COL),
              col(s"$c.${GraphFrame.DST}").alias(DST_ID_COL),
              col(s"$c.${GraphFrame.ATTR}").alias(ATTR_COL)))
          .toIndexedSeq: _*)

      resDF
        .select(verticesExpr.alias(VERTICES_COL), edgesExpr.alias(EDGES_COL))
        .as[GraphPath[VD, ED]]
    }
  }
}
