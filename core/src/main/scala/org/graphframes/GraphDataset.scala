package org.graphframes

import org.apache.spark.graphx.VertexId
import org.apache.spark.sql.{Dataset, Encoder, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.graphframes.lib.{GraphEdge, GraphVertex}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag
class GraphDataset[VD : Encoder : ClassTag : TypeTag, ED : Encoder : ClassTag : TypeTag] private(
                                                          private val _vertices: Dataset[GraphVertex[VD]],
                                                          private val _edges: Dataset[GraphEdge[ED]])(@transient implicit val spark: SparkSession) extends Logging with Serializable {

    import spark.implicits._

    override def toString: String = {
        ""
    }

    /**
     * Persist the dataframe representation of vertices and edges of the graph with the default
     * storage level.
     */
    def cache(): this.type = {
        persist()
    }

    /**
     * Persist the dataframe representation of vertices and edges of the graph with the default
     * storage level.
     */
    def persist(): this.type = {
        vertices.persist()
        edges.persist()
        this
    }

    /**
     * Persist the dataframe representation of vertices and edges of the graph with the given
     * storage level.
     * @param newLevel  One of: `MEMORY_ONLY`, `MEMORY_AND_DISK`, `MEMORY_ONLY_SER`,
     * `MEMORY_AND_DISK_SER`, `DISK_ONLY`, `MEMORY_ONLY_2`, `MEMORY_AND_DISK_2`, etc..
     */
    def persist(newLevel: StorageLevel): this.type = {
        vertices.persist(newLevel)
        edges.persist(newLevel)
        this
    }

    /**
     * Mark the dataframe representation of vertices and edges of the graph as non-persistent, and
     * remove all blocks for it from memory and disk.
     */
    def unpersist(): this.type = {
        vertices.unpersist()
        edges.unpersist()
        this
    }

    /**
     * Mark the dataframe representation of vertices and edges of the graph as non-persistent, and
     * remove all blocks for it from memory and disk.
     * @param blocking  Whether to block until all blocks are deleted.
     */
    def unpersist(blocking: Boolean): this.type = {
        vertices.unpersist(blocking)
        edges.unpersist(blocking)
        this
    }

    // ============== Basic structural methods ============

    /**
     * The dataframe representation of the vertices of the graph.
     *
     * It contains a column called [[GraphFrame.ID]] with the id of the vertex,
     * and various other user-defined attributes with other attributes.
     *
     * The order of the columns is available in [[vertexColumns]].
     *
     * @group structure
     */
    def vertices: Dataset[GraphVertex[VD]] = {
        if (_vertices == null) {
        throw new Exception("You cannot use GraphFrame objects within a Spark closure")
        }
        _vertices
    }

    /**
     * The dataframe representation of the edges of the graph.
     *
     * It contains two columns called [[GraphFrame.SRC]] and [[GraphFrame.DST]] that contain
     * the ids of the source vertex and the destination vertex of each edge, respectively.
     * It may also contain various other columns with user-defined attributes for each edge.
     *
     * For symmetric graphs, both pairs src -> dst and dst -> src are present with the same
     * attributes for each pair.
     *
     * The order of the columns is available in [[edgeColumns]].
     *
     * @group structure
     */
    def edges: Dataset[GraphEdge[ED]] = {
        if (_edges == null) {
        throw new Exception("You cannot use GraphFrame objects within a Spark closure")
        }
        _edges
    }

    @transient lazy val outDegrees: Dataset[(VertexId, Int)] = {
        _edges
            .groupByKey((e: GraphEdge[ED]) => e.srcId)
            .mapValues(_ => 1)
            .reduceGroups(_ + _)
    }    

    @transient lazy val inDegrees: Dataset[(VertexId, Int)] = {
        _edges
            .groupByKey((e: GraphEdge[ED]) => e.dstId)
            .mapValues(_ => 1)
            .reduceGroups(_ + _)
    }

    @transient lazy val degrees: Dataset[(VertexId, Int)] = {
        _edges
            .flatMap((e: GraphEdge[ED]) => Array(e.dstId, e.srcId))
            .groupByKey((id: VertexId) => id)
            .mapValues(_ => 1)
            .reduceGroups(_ + _)
    }

    def filterVertices(condition: GraphVertex[VD] => Boolean): GraphDataset[VD, ED] = {
      import spark.implicits._
      val vv: Dataset[GraphVertex[VD]] = vertices.filter(condition)

      val ee: Dataset[GraphEdge[ED]] = edges.groupByKey(e => e.srcId)
        .cogroup(vv.groupByKey(v => v.id)) { case (vid, edgesIter, vvIter) => {
          edgesIter
        }
        }
        .groupByKey(e => e.dstId)
        .cogroup(vv.groupByKey(v => v.id)) { case (vid, edgesIter, vvIter) => {
          edgesIter
        }
        }
        
        GraphDataset(vv, ee)
    }
  
    def filterEdges(condition: (GraphEdge[ED]) => Boolean): GraphDataset[VD, ED] = {
      val vv: Dataset[GraphVertex[VD]] = vertices
      val ee: Dataset[GraphEdge[ED]] = edges.filter(condition)
      GraphDataset(vv, ee)
    }
  
    def dropIsolatedVertices(): GraphDataset[VD, ED] = {
      import spark.implicits._
      val ee: Dataset[GraphEdge[ED]] = edges
      val e1 = ee.flatMap(edge => List(edge.dstId, edge.srcId)
              .map(id => (id, edge.attr))
          )
          .groupByKey(e => e._1)

      val vv: Dataset[GraphVertex[VD]] = vertices
        .groupByKey(v => v.id)
        .cogroup(e1) { case (vid, vertIter, edgeIter) => {
          vertIter
        }
        }

      GraphDataset(vv, ee)
    }
}

object GraphDataset extends Serializable with Logging {

    def apply[VD : Encoder: ClassTag : TypeTag, ED : Encoder : ClassTag : TypeTag](vertices: Dataset[GraphVertex[VD]], edges: Dataset[GraphEdge[ED]])(implicit spark: SparkSession): GraphDataset[VD, ED] = {
      new GraphDataset(vertices, edges)
    }
    
    def fromEdges[ED : Encoder : ClassTag : TypeTag, VD : Encoder : ClassTag : TypeTag](edges: Dataset[GraphEdge[ED]], defaultVertexAttribute: VD = null.asInstanceOf[VD])(implicit spark: SparkSession): GraphDataset[VD, ED] = {
      import spark.implicits._
      val srcs = edges.map(e => e.srcId)
      val dsts = edges.map(e => e.dstId)
      val vertices: Dataset[GraphVertex[VD]] = srcs.unionAll(dsts)
          .distinct()
          .map((id: VertexId) => GraphVertex(id, defaultVertexAttribute))
      apply(vertices, edges)
    }
}