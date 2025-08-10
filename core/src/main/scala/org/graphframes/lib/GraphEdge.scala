package org.graphframes.lib

import org.apache.spark.sql.{Encoder, Encoders}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

trait GraphEdgeBase[ED] {
  def srcId: VertexId
  def dstId: VertexId
  def attr: ED
}

case class GraphEdge[ED](var srcId: VertexId = 0,
                                   var dstId: VertexId = 0,
                                   var attr: ED = null.asInstanceOf[ED])
  extends Serializable with GraphEdgeBase[ED]

object GraphEdge {
  implicit def encoder[ED: Encoder : ClassTag : TypeTag]: Encoder[GraphEdge[ED]] =
    Encoders.product[GraphEdge[ED]]
}