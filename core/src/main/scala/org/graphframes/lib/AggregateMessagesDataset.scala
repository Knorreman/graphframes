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

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.expressions.Aggregator
import org.graphframes.GraphDataset
import org.graphframes.Logging

import scala.collection.mutable.ListBuffer

/**
 * Typed message aggregation primitive for [[org.graphframes.GraphDataset]].
 *
 * For each triplet `(src, edge, dst)` in [[org.graphframes.GraphDataset.triplets]], messages of
 * type `M` can be sent to the source vertex, destination vertex, or arbitrary vertices, and then
 * combined per recipient vertex using `agg`.
 *
 * @example
 *   Compute the sum of source vertex attributes received by each destination vertex:
 *   {{{
 *   val sums: Dataset[(VertexId, Int)] = graph
 *     .aggregateMessages[Int]
 *     .sendToDst(t => t.srcAttr)
 *     .agg(_ + _)
 *   }}}
 *
 * @tparam VD
 *   the vertex attribute type
 * @tparam ED
 *   the edge attribute type
 * @tparam M
 *   the message type
 */
class AggregateMessagesDataset[VD, ED, M: Encoder] private[graphframes] (
    private val graph: GraphDataset[VD, ED])
    extends Arguments
    with Serializable {

  private val senders = ListBuffer.empty[GraphTriplet[VD, ED] => Iterator[(VertexId, M)]]

  /**
   * Sends a message to the source vertex of each triplet.
   */
  def sendToSrc(f: GraphTriplet[VD, ED] => M): this.type = {
    senders += (t => Iterator.single((t.srcId, f(t))))
    this
  }

  /**
   * Conditionally sends a message to the source vertex of each triplet (`None` sends no message).
   */
  def sendToSrcOpt(f: GraphTriplet[VD, ED] => Option[M]): this.type = {
    senders += (t => f(t).iterator.map(m => (t.srcId, m)))
    this
  }

  /**
   * Sends a message to the destination vertex of each triplet.
   */
  def sendToDst(f: GraphTriplet[VD, ED] => M): this.type = {
    senders += (t => Iterator.single((t.dstId, f(t))))
    this
  }

  /**
   * Conditionally sends a message to the destination vertex of each triplet (`None` sends no
   * message).
   */
  def sendToDstOpt(f: GraphTriplet[VD, ED] => Option[M]): this.type = {
    senders += (t => f(t).iterator.map(m => (t.dstId, m)))
    this
  }

  /**
   * Sends zero or more `(targetVertexId, message)` pairs for each triplet.
   */
  def sendMsg(f: GraphTriplet[VD, ED] => Iterator[(VertexId, M)]): this.type = {
    senders += f
    this
  }

  private def buildRawMessages(): Dataset[(VertexId, M)] = {
    require(
      senders.nonEmpty,
      "To run GraphDataset.aggregateMessages, at least one message sender must be configured " +
        "using sendToSrc, sendToDst, or sendMsg.")
    implicit val msgPairEncoder: Encoder[(VertexId, M)] =
      Encoders.tuple(Encoders.scalaLong, implicitly[Encoder[M]])
    val senderFns = senders.toVector
    graph.triplets.flatMap { t =>
      senderFns.iterator.flatMap(fn => fn(t))
    }
  }

  /**
   * Runs the aggregation by reducing messages sent to each vertex with `mergeMsg`.
   *
   * Returns a `Dataset[(VertexId, M)]` with one entry per vertex that received at least one
   * message.
   */
  def agg(mergeMsg: (M, M) => M): Dataset[(VertexId, M)] = {
    implicit val vertexIdEncoder: Encoder[VertexId] = Encoders.scalaLong
    buildRawMessages()
      .groupByKey(_._1)
      .mapValues(_._2)
      .reduceGroups(mergeMsg)
  }

  /**
   * Runs the aggregation using a typed Spark `org.apache.spark.sql.expressions.Aggregator`,
   * allowing the output type `R` to differ from the message type `M` (for example, computing an
   * average over integer messages).
   */
  def agg[R](aggregator: Aggregator[M, ?, R]): Dataset[(VertexId, R)] = {
    implicit val vertexIdEncoder: Encoder[VertexId] = Encoders.scalaLong
    buildRawMessages()
      .groupByKey(_._1)
      .mapValues(_._2)
      .agg(aggregator.toColumn)
  }
}

object AggregateMessagesDataset extends Logging {

  /**
   * Aggregates messages from neighboring vertices and edges in a single pass.
   */
  def run[VD, ED, M: Encoder](
      graph: GraphDataset[VD, ED],
      sendMsg: GraphTriplet[VD, ED] => Iterator[(VertexId, M)],
      mergeMsg: (M, M) => M): Dataset[(VertexId, M)] = {
    graph.aggregateMessages[M].sendMsg(sendMsg).agg(mergeMsg)
  }
}
