package org.graphframes.lib

import org.apache.spark.sql.{Encoder, Encoders}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

trait GraphVertexBase[VD] {
    def id: VertexId
    def attr: VD
}

case class GraphVertex[VD](id: VertexId, attr: VD) extends Serializable with GraphVertexBase[VD]

object GraphVertex {
  implicit def encoder[VD: Encoder: ClassTag : TypeTag]: Encoder[GraphVertex[VD]] =
    Encoders.product[GraphVertex[VD]]
}