package org.graphframes.lib

import org.apache.spark.sql.{Dataset, Encoder, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.graphframes.{GraphDataset, Logging}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

class ConnectedComponentsDataset[VD : Encoder : ClassTag : TypeTag, ED : Encoder : ClassTag : TypeTag] private[graphframes] (
    private val graph: GraphDataset[ED, VD])(@transient implicit val spark : SparkSession) extends Arguments with Logging {

    private var broadcastThreshold: Int = 1000000

  /**
   * Sets broadcast threshold in propagating component assignments (default: 1000000).
   * If a node degree is greater than this threshold at some iteration, its component assignment
   * will be collected and then broadcasted back to propagate the assignment to its neighbors.
   * Otherwise, the assignment propagation is done by a normal Spark join.
   * This parameter is only used when the algorithm is set to "graphframes".
   */
  def setBroadcastThreshold(value: Int): this.type = {
    require(value >= 0, s"Broadcast threshold must be non-negative but got $value.")
    broadcastThreshold = value
    this
  }

  /**
   * Gets broadcast threshold in propagating component assignment.
   * @see [[org.graphframes.lib.ConnectedComponents.setBroadcastThreshold]]
   */
  def getBroadcastThreshold: Int = broadcastThreshold
  private var checkpointInterval: Int = 2

  def setCheckpointInterval(value: Int): this.type = {
    if (value <= 0 || value > 2) {
      logWarn(
        s"Set checkpointInterval to $value. This would blow up the query plan and hang the " +
          "driver for large graphs.")
    }
    checkpointInterval = value
    this
  }

  // python-friendly setter
  private[graphframes] def setCheckpointInterval(value: java.lang.Integer): this.type = {
    setCheckpointInterval(value.toInt)
  }

  /**
   * Gets checkpoint interval.
   * @see [[org.graphframes.lib.ConnectedComponents.setCheckpointInterval]]
   */
  def getCheckpointInterval: Int = checkpointInterval

  private var intermediateStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK

  /**
   * Sets storage level for intermediate datasets that require multiple passes (default: ``MEMORY_AND_DISK``).
   */
  def setIntermediateStorageLevel(value: StorageLevel): this.type = {
    intermediateStorageLevel = value
    this
  }

  /**
   * Gets storage level for intermediate datasets that require multiple passes.
   */
  def getIntermediateStorageLevel: StorageLevel = intermediateStorageLevel
  /**
   * Runs the algorithm.
   */
  def run(): Dataset[ConnectedComponentResult] = {
    ConnectedComponentsDataset.run(graph,
      broadcastThreshold = broadcastThreshold,
      checkpointInterval = checkpointInterval,
      intermediateStorageLevel = intermediateStorageLevel)
  }
}

private object ConnectedComponentsDataset extends Logging {

    private def prepare[VD : Encoder: ClassTag : TypeTag, ED : Encoder: ClassTag : TypeTag](graph: GraphDataset[VD, ED])(implicit spark: SparkSession): GraphDataset[VD, ED] = {
      import spark.implicits._
      val ee = graph.edges.map {
          case e @ GraphEdge(src: Long, dst: Long, attr: ED) => {
              if (src > dst) GraphEdge(dst, src, attr)
              e
          }
      }.distinct()

      GraphDataset(graph.vertices, ee)
    }

    private def symmetrize[VD : Encoder: ClassTag : TypeTag, ED : Encoder: ClassTag : TypeTag](edges: Dataset[GraphEdge[ED]]): Dataset[GraphEdge[ED]] = {
        ???
    }

    private def run[VD : Encoder: ClassTag : TypeTag, ED : Encoder: ClassTag : TypeTag](graph: GraphDataset[ED, VD],
        broadcastThreshold: Int,
        checkpointInterval: Int,
        intermediateStorageLevel: StorageLevel): Dataset[ConnectedComponentResult] = {
        ???
    }
}

case class ConnectedComponentResult(id: VertexId, connectedComponent: Long) extends GraphVertexBase[Long] {
    def attr: Long = connectedComponent
    def toVertex: GraphVertex[Long] = GraphVertex(id, connectedComponent)
}