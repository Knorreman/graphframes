package org.graphframes.lib

import org.apache.spark.graphx.{EdgeDirection, EdgeTriplet}
import org.apache.spark.sql.Encoder
import org.graphframes.{GraphDataset, Logging}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

class PregelGraphDataset[VD : Encoder: ClassTag : TypeTag, ED : Encoder: ClassTag : TypeTag] private[graphframes](
    private val graph: GraphDataset[ED, VD]) extends Arguments with Logging {  

  
  private var maxIter: Int = 10
  private var checkpointInterval = 2

  /** Sets the max number of iterations (default: 10). */
  def setMaxIter(value: Int): this.type = {
    maxIter = value
    this
  }

  /**
   * Sets the number of iterations between two checkpoints (default: 2).
   *
   * This is an advanced control to balance query plan optimization and checkpoint data I/O cost.
   * In most cases, you should keep the default value.
   *
   * Checkpoint is disabled if this is set to 0.
   */
  def setCheckpointInterval(value: Int): this.type = {
    checkpointInterval = value
    this
  }

}


object PregelGraphDataset {

    def run[VD : Encoder: ClassTag : TypeTag, ED : Encoder: ClassTag : TypeTag, A : Encoder: ClassTag : TypeTag](
        graph: GraphDataset[VD, ED],
        initialMsg: A,       
        maxIterations: Int = Int.MaxValue,
        checkpointInterval: Int = 0,
        activeDirection: EdgeDirection = EdgeDirection.Either,
        vprog: (VertexId, VD, A) => VD,
        sendMsg: EdgeTriplet[VD, ED] => Iterator[(VertexId, A)],
        mergeMsg: (A, A) => A
    ): GraphDataset[VD, ED] = {
        ???
    }
}