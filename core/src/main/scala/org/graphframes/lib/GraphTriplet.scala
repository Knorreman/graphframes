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

import org.apache.spark.graphframes.graphx.EdgeTriplet
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders

import scala.reflect.runtime.universe.TypeTag

/**
 * A typed edge triplet representing an edge along with the attributes of its source and
 * destination vertices.
 *
 * @param srcId
 *   the source vertex identifier
 * @param dstId
 *   the destination vertex identifier
 * @param srcAttr
 *   the source vertex attribute
 * @param dstAttr
 *   the destination vertex attribute
 * @param attr
 *   the edge attribute
 * @tparam VD
 *   the vertex attribute type
 * @tparam ED
 *   the edge attribute type
 */
case class GraphTriplet[VD, ED](
    srcId: VertexId,
    dstId: VertexId,
    srcAttr: VD,
    dstAttr: VD,
    attr: ED)
    extends Serializable
    with GraphEdgeBase[ED] {

  /** The source vertex. */
  def src: GraphVertex[VD] = GraphVertex(srcId, srcAttr)

  /** The destination vertex. */
  def dst: GraphVertex[VD] = GraphVertex(dstId, dstAttr)

  /** The edge without vertex attributes. */
  def edge: GraphEdge[ED] = GraphEdge(srcId, dstId, attr)

  /**
   * Given one vertex in the edge, return the attribute of the other vertex.
   */
  def otherVertexAttr(vid: VertexId): VD =
    if (srcId == vid) dstAttr
    else {
      assert(dstId == vid)
      srcAttr
    }

  /**
   * Given one vertex in the edge, return its attribute.
   */
  def vertexAttr(vid: VertexId): VD =
    if (srcId == vid) srcAttr
    else {
      assert(dstId == vid)
      dstAttr
    }

  /** Converts this triplet to a GraphX `EdgeTriplet`. */
  def toEdgeTriplet: EdgeTriplet[VD, ED] = {
    val et = new EdgeTriplet[VD, ED]
    et.srcId = srcId
    et.dstId = dstId
    et.srcAttr = srcAttr
    et.dstAttr = dstAttr
    et.attr = attr
    et
  }

  /** Converts this triplet to a nested tuple `((srcId, srcAttr), (dstId, dstAttr), attr)`. */
  def toTuple: ((VertexId, VD), (VertexId, VD), ED) = ((srcId, srcAttr), (dstId, dstAttr), attr)
}

object GraphTriplet {
  implicit def encoder[VD: TypeTag, ED: TypeTag]: Encoder[GraphTriplet[VD, ED]] =
    Encoders.product[GraphTriplet[VD, ED]]
}
