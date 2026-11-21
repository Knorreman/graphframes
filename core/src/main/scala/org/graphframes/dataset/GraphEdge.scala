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

package org.graphframes.dataset

import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders

import scala.reflect.runtime.universe.TypeTag

/**
 * Base trait for typed graph edges.
 *
 * @tparam ED
 *   the edge attribute type
 */
trait GraphEdgeBase[ED] {
  def srcId: VertexId
  def dstId: VertexId
  def attr: ED
}

/**
 * A typed directed edge in a [[org.graphframes.dataset.GraphDataset]].
 *
 * @param srcId
 *   the source vertex identifier
 * @param dstId
 *   the destination vertex identifier
 * @param attr
 *   the edge attribute
 * @tparam ED
 *   the edge attribute type
 */
case class GraphEdge[ED](
    srcId: VertexId = 0L,
    dstId: VertexId = 0L,
    attr: ED = null.asInstanceOf[ED])
    extends Serializable
    with GraphEdgeBase[ED]

object GraphEdge {
  implicit def encoder[ED: TypeTag]: Encoder[GraphEdge[ED]] =
    Encoders.product[GraphEdge[ED]]
}
