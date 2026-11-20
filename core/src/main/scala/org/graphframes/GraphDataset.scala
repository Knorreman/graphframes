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

package org.graphframes

import org.apache.spark.graphframes.graphx.Edge
import org.apache.spark.graphframes.graphx.Graph
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.count
import org.apache.spark.sql.functions.countDistinct
import org.apache.spark.sql.functions.explode
import org.apache.spark.storage.StorageLevel
import org.graphframes.lib.AggregateMessagesDataset
import org.graphframes.lib.ConnectedComponentsDataset
import org.graphframes.lib.GraphEdge
import org.graphframes.lib.GraphTriplet
import org.graphframes.lib.GraphVertex
import org.graphframes.lib.PregelGraphDataset
import org.graphframes.lib.VertexId

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

/**
 * A typed representation of a graph backed by Spark `Dataset`s of
 * [[org.graphframes.lib.GraphVertex]] and [[org.graphframes.lib.GraphEdge]].
 *
 * @tparam VD
 *   the vertex attribute type
 * @tparam ED
 *   the edge attribute type
 *
 * @groupname structure Structure information
 * @groupname conversions Conversions
 * @groupname stdlib Standard graph algorithms
 * @groupname subgraph Subgraph selection and transformations
 * @groupname degree Graph topology
 * @groupname utils Utility Methods
 */
class GraphDataset[VD: TypeTag, ED: TypeTag] private (
    @transient private val _vertices: Dataset[GraphVertex[VD]],
    @transient private val _edges: Dataset[GraphEdge[ED]])
    extends Logging
    with Serializable {

  private[graphframes] def spark: SparkSession = vertices.sparkSession

  /**
   * Return a string representation of the GraphDataset.
   *
   * @group utils
   */
  override def toString: String = {
    s"GraphDataset(v:${vertices.toString}, e:${edges.toString})"
  }

  /**
   * Persist the dataset representation of vertices and edges of the graph with the default
   * storage level.
   *
   * @group utils
   */
  def cache(): this.type = {
    persist()
  }

  /**
   * Persist the dataset representation of vertices and edges of the graph with the default
   * storage level.
   *
   * @group utils
   */
  def persist(): this.type = {
    vertices.persist()
    edges.persist()
    this
  }

  /**
   * Persist the dataset representation of vertices and edges of the graph with the given storage
   * level.
   *
   * @param newLevel
   *   One of: `MEMORY_ONLY`, `MEMORY_AND_DISK`, `MEMORY_ONLY_SER`, `MEMORY_AND_DISK_SER`,
   *   `DISK_ONLY`, `MEMORY_ONLY_2`, `MEMORY_AND_DISK_2`, etc.
   * @group utils
   */
  def persist(newLevel: StorageLevel): this.type = {
    vertices.persist(newLevel)
    edges.persist(newLevel)
    this
  }

  /**
   * Mark the dataset representation of vertices and edges of the graph as non-persistent, and
   * remove all blocks for it from memory and disk.
   *
   * @group utils
   */
  def unpersist(): this.type = {
    vertices.unpersist()
    edges.unpersist()
    this
  }

  /**
   * Mark the dataset representation of vertices and edges of the graph as non-persistent, and
   * remove all blocks for it from memory and disk.
   *
   * @param blocking
   *   Whether to block until all blocks are deleted.
   * @group utils
   */
  def unpersist(blocking: Boolean): this.type = {
    vertices.unpersist(blocking)
    edges.unpersist(blocking)
    this
  }

  /**
   * Validates the consistency and integrity of a graph by performing checks on the vertices and
   * edges.
   *
   * @throws InvalidGraphException
   *   if there are any inconsistencies in the graph, such as duplicate vertices or edges
   *   referencing vertices not present in the vertices dataset.
   * @group utils
   */
  def validate(): Unit =
    validate(checkVertices = true, intermediateStorageLevel = StorageLevel.MEMORY_AND_DISK)

  /**
   * Validates the consistency and integrity of a graph by performing checks on the vertices and
   * edges.
   *
   * @param checkVertices
   *   whether to verify that all edge endpoints exist in the vertices dataset.
   * @param intermediateStorageLevel
   *   storage level used for intermediate datasets during validation.
   * @throws InvalidGraphException
   *   if there are any inconsistencies in the graph.
   * @group utils
   */
  def validate(checkVertices: Boolean, intermediateStorageLevel: StorageLevel): Unit = {
    val persistedVertices = vertices.persist(intermediateStorageLevel)
    try {
      val countDistinctVertices = persistedVertices.select(countDistinct("id")).first().getLong(0)
      val verticesCount = persistedVertices.count()
      if (countDistinctVertices != verticesCount) {
        throw new InvalidGraphException(
          s"Graph contains (${verticesCount - countDistinctVertices}) duplicate vertices.")
      }
      if (checkVertices) {
        val verticesSetFromEdges = edges
          .select(col("srcId").alias("id"))
          .union(edges.select(col("dstId").alias("id")))
          .distinct()
          .persist(intermediateStorageLevel)
        try {
          val countVerticesFromEdges = verticesSetFromEdges.count()
          if (countVerticesFromEdges > countDistinctVertices) {
            throw new InvalidGraphException(
              s"Graph is inconsistent: edges has $countVerticesFromEdges " +
                s"vertices, but vertices has $countDistinctVertices vertices.")
          }
          val countOfBadVertices = verticesSetFromEdges
            .join(persistedVertices, Seq("id"), "left_anti")
            .count()
          if (countOfBadVertices > 0) {
            throw new InvalidGraphException(
              "Vertices Dataset does not contain all edges srcId/dstId. " +
                s"Found $countOfBadVertices edges srcId/dstId that are not in the vertices Dataset.")
          }
        } finally {
          val _ = verticesSetFromEdges.unpersist()
        }
      }
    } finally {
      val _ = persistedVertices.unpersist()
    }
  }

  /**
   * Converts the directed graph into an undirected graph by adding a reverse edge `(dstId, srcId,
   * attr)` for every edge `(srcId, dstId, attr)`.
   *
   * @group utils
   */
  def asUndirected(): GraphDataset[VD, ED] = {
    val reversed = edges.map(e => GraphEdge(e.dstId, e.srcId, e.attr))
    GraphDataset(vertices, edges.union(reversed))
  }

  /**
   * Reverses the direction of all edges in the graph.
   *
   * @group utils
   */
  def asReversed(): GraphDataset[VD, ED] = {
    val reversed = edges.map(e => GraphEdge(e.dstId, e.srcId, e.attr))
    GraphDataset(vertices, reversed)
  }

  // ============== Basic structural methods ============

  /**
   * The typed dataset representation of the vertices of the graph.
   *
   * @group structure
   */
  def vertices: Dataset[GraphVertex[VD]] = {
    if (_vertices == null) {
      throw new Exception("You cannot use GraphDataset objects within a Spark closure")
    }
    _vertices
  }

  /**
   * The typed dataset representation of the edges of the graph.
   *
   * @group structure
   */
  def edges: Dataset[GraphEdge[ED]] = {
    if (_edges == null) {
      throw new Exception("You cannot use GraphDataset objects within a Spark closure")
    }
    _edges
  }

  /**
   * Returns typed triplets `(src, edge, dst)` for all edges in the graph whose source and
   * destination vertices exist in [[vertices]].
   *
   * @group structure
   */
  @transient lazy val triplets: Dataset[GraphTriplet[VD, ED]] = {
    val srcV = vertices.toDF("srcId", "srcAttr")
    val dstV = vertices.toDF("dstId", "dstAttr")
    edges
      .toDF("srcId", "dstId", "attr")
      .join(srcV, Seq("srcId"), "inner")
      .join(dstV, Seq("dstId"), "inner")
      .select(col("srcId"), col("dstId"), col("srcAttr"), col("dstAttr"), col("attr"))
      .as[GraphTriplet[VD, ED]]
  }

  // ============== Conversions ============

  /**
   * Converts this [[GraphDataset]] into an untyped [[GraphFrame]].
   *
   * The resulting vertices DataFrame has columns `id` and `attr`, and the edges DataFrame has
   * columns `src`, `dst`, and `attr`.
   *
   * @group conversions
   */
  def toGraphFrame: GraphFrame = {
    val vDF = vertices.toDF(GraphFrame.ID, GraphFrame.ATTR)
    val eDF = edges.toDF(GraphFrame.SRC, GraphFrame.DST, GraphFrame.ATTR)
    GraphFrame(vDF, eDF)
  }

  /**
   * Converts this [[GraphDataset]] into a strongly typed GraphX `Graph[VD, ED]`.
   *
   * @group conversions
   */
  def toGraphX(implicit vdCt: ClassTag[VD], edCt: ClassTag[ED]): Graph[VD, ED] = {
    val vRDD = vertices.rdd.map(v => (v.id, v.attr))
    val eRDD = edges.rdd.map(e => Edge(e.srcId, e.dstId, e.attr))
    Graph[VD, ED](vRDD, eRDD)
  }

  // ============== Degree metrics ============

  /**
   * The out-degree of each vertex in the graph. Vertices with 0 out-edges are not returned.
   *
   * @group degree
   */
  @transient lazy val outDegrees: Dataset[(VertexId, Int)] = {
    val session = spark
    import session.implicits.*
    edges
      .groupBy(col("srcId"))
      .agg(count("*").cast("int"))
      .as[(VertexId, Int)]
  }

  /**
   * The in-degree of each vertex in the graph. Vertices with 0 in-edges are not returned.
   *
   * @group degree
   */
  @transient lazy val inDegrees: Dataset[(VertexId, Int)] = {
    val session = spark
    import session.implicits.*
    edges
      .groupBy(col("dstId"))
      .agg(count("*").cast("int"))
      .as[(VertexId, Int)]
  }

  /**
   * The total degree (in + out) of each vertex in the graph. Vertices with 0 edges are not
   * returned.
   *
   * @group degree
   */
  @transient lazy val degrees: Dataset[(VertexId, Int)] = {
    val session = spark
    import session.implicits.*
    edges
      .select(explode(array(col("srcId"), col("dstId"))).alias("id"))
      .groupBy(col("id"))
      .agg(count("*").cast("int"))
      .as[(VertexId, Int)]
  }

  // ============== Subgraph selection and transformations ============

  /**
   * Filter the vertices according to a typed predicate, removing edges containing any dropped
   * vertices.
   *
   * @group subgraph
   */
  def filterVertices(condition: GraphVertex[VD] => Boolean): GraphDataset[VD, ED] = {
    val vv: Dataset[GraphVertex[VD]] = vertices.filter(condition)
    val vvIds = vv.select(col("id"))
    val ee: Dataset[GraphEdge[ED]] = edges
      .join(vvIds, col("srcId") === vvIds("id"), "left_semi")
      .join(vvIds, col("dstId") === vvIds("id"), "left_semi")
      .as[GraphEdge[ED]]
    GraphDataset(vv, ee)
  }

  /**
   * Filter the edges according to a typed predicate, keeping all vertices.
   *
   * @group subgraph
   */
  def filterEdges(condition: GraphEdge[ED] => Boolean): GraphDataset[VD, ED] = {
    val ee: Dataset[GraphEdge[ED]] = edges.filter(condition)
    GraphDataset(vertices, ee)
  }

  /**
   * Restrict the graph to only the vertices and edges satisfying the predicates. Edges whose
   * source or destination vertices do not satisfy `vpred` are also removed.
   *
   * @group subgraph
   */
  def subgraph(
      epred: GraphEdge[ED] => Boolean = _ => true,
      vpred: GraphVertex[VD] => Boolean = _ => true): GraphDataset[VD, ED] = {
    filterVertices(vpred).filterEdges(epred)
  }

  /**
   * Drop isolated vertices (vertices not incident to any edge).
   *
   * @group subgraph
   */
  def dropIsolatedVertices(): GraphDataset[VD, ED] = {
    val ee: Dataset[GraphEdge[ED]] = edges
    val activeIds = ee.select(explode(array(col("srcId"), col("dstId"))).alias("id"))
    val vv: Dataset[GraphVertex[VD]] = vertices
      .join(activeIds, Seq("id"), "left_semi")
      .as[GraphVertex[VD]]
    GraphDataset(vv, ee)
  }

  /**
   * Transform each vertex attribute using the given function.
   *
   * @group subgraph
   */
  def mapVertices[VD2: TypeTag](f: (VertexId, VD) => VD2): GraphDataset[VD2, ED] = {
    val vv: Dataset[GraphVertex[VD2]] = vertices.map(v => GraphVertex(v.id, f(v.id, v.attr)))
    GraphDataset(vv, edges)
  }

  /**
   * Transform each edge attribute using the given function.
   *
   * @group subgraph
   */
  def mapEdges[ED2: TypeTag](f: GraphEdge[ED] => ED2): GraphDataset[VD, ED2] = {
    val ee: Dataset[GraphEdge[ED2]] = edges.map(e => GraphEdge(e.srcId, e.dstId, f(e)))
    GraphDataset(vertices, ee)
  }

  /**
   * Transform each edge attribute using the full triplet (source vertex, edge, destination
   * vertex).
   *
   * @group subgraph
   */
  def mapTriplets[ED2: TypeTag](f: GraphTriplet[VD, ED] => ED2): GraphDataset[VD, ED2] = {
    val ee: Dataset[GraphEdge[ED2]] = triplets.map(t => GraphEdge(t.srcId, t.dstId, f(t)))
    GraphDataset(vertices, ee)
  }

  // ============== Standard graph algorithms ============

  /**
   * Typed message aggregation builder for sending and aggregating messages across graph triplets.
   *
   * See [[org.graphframes.lib.AggregateMessagesDataset]] for more details.
   *
   * @tparam M
   *   the message type
   * @group stdlib
   */
  def aggregateMessages[M: Encoder]: AggregateMessagesDataset[VD, ED, M] =
    new AggregateMessagesDataset[VD, ED, M](this)

  /**
   * Connected components algorithm.
   *
   * See [[org.graphframes.lib.ConnectedComponentsDataset]] for more details.
   *
   * @group stdlib
   */
  def connectedComponents: ConnectedComponentsDataset[VD, ED] =
    new ConnectedComponentsDataset[VD, ED](this)

  /**
   * Typed Pregel algorithm.
   *
   * See [[org.graphframes.lib.PregelGraphDataset]] for more details.
   *
   * @group stdlib
   */
  def pregel: PregelGraphDataset[VD, ED] =
    new PregelGraphDataset[VD, ED](this)
}

object GraphDataset extends Serializable with Logging {

  /**
   * Construct a [[GraphDataset]] from typed vertex and edge datasets.
   */
  def apply[VD: TypeTag, ED: TypeTag](
      vertices: Dataset[GraphVertex[VD]],
      edges: Dataset[GraphEdge[ED]]): GraphDataset[VD, ED] = {
    new GraphDataset[VD, ED](vertices, edges)
  }

  /**
   * Construct a [[GraphDataset]] from an edge dataset, inferring the vertex set from the distinct
   * edge endpoints and assigning each vertex `defaultVertexAttribute`.
   */
  def fromEdges[ED: TypeTag, VD: TypeTag](
      edges: Dataset[GraphEdge[ED]],
      defaultVertexAttribute: VD = null.asInstanceOf[VD]): GraphDataset[VD, ED] = {
    val session = edges.sparkSession
    import session.implicits.*
    val srcs = edges.map(e => e.srcId)
    val dsts = edges.map(e => e.dstId)
    val vertices: Dataset[GraphVertex[VD]] = srcs
      .union(dsts)
      .distinct()
      .map((id: VertexId) => GraphVertex(id, defaultVertexAttribute))
    apply(vertices, edges)
  }

  /**
   * Construct a [[GraphDataset]] from an existing [[GraphFrame]].
   *
   * The vertices DataFrame must contain `id` (castable to `Long`) and `attr` columns, and the
   * edges DataFrame must contain `src`, `dst` (castable to `Long`), and `attr` columns.
   */
  def fromGraphFrame[VD: TypeTag, ED: TypeTag](graph: GraphFrame): GraphDataset[VD, ED] = {
    val vv = graph.vertices
      .select(col(GraphFrame.ID).cast("long").alias("id"), col(GraphFrame.ATTR).alias("attr"))
      .as[GraphVertex[VD]]
    val ee = graph.edges
      .select(
        col(GraphFrame.SRC).cast("long").alias("srcId"),
        col(GraphFrame.DST).cast("long").alias("dstId"),
        col(GraphFrame.ATTR).alias("attr"))
      .as[GraphEdge[ED]]
    apply(vv, ee)
  }
}
