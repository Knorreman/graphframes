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
 * Base trait for typed graph vertices.
 *
 * @tparam VD
 *   the vertex attribute type
 */
trait GraphVertexBase[VD] {
  def id: VertexId
  def attr: VD
}

/**
 * A typed vertex in a [[org.graphframes.dataset.GraphDataset]].
 *
 * @param id
 *   the unique vertex identifier
 * @param attr
 *   the vertex attribute
 * @tparam VD
 *   the vertex attribute type
 */
case class GraphVertex[VD](id: VertexId, attr: VD) extends Serializable with GraphVertexBase[VD]

object GraphVertex {
  implicit def encoder[VD: TypeTag]: Encoder[GraphVertex[VD]] =
    Encoders.product[GraphVertex[VD]]
}
