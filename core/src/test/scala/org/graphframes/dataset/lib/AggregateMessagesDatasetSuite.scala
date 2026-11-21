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

package org.graphframes.dataset.lib;

import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.expressions.Aggregator
import org.graphframes.GraphFrameTestSparkContext
import org.graphframes.SparkFunSuite
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphEdge
import org.graphframes.dataset.GraphVertex
import org.scalactic.Tolerance.*

case class UserProfile(name: String, age: Int)
case class LinkMeta(kind: String, bonus: Int)
case class AgeSummary(totalAge: Int, count: Int)

private object AvgIntAggregator extends Aggregator[Int, (Long, Long), Double] {
  override def zero: (Long, Long) = (0L, 0L)
  override def reduce(b: (Long, Long), a: Int): (Long, Long) = (b._1 + a, b._2 + 1L)
  override def merge(b1: (Long, Long), b2: (Long, Long)): (Long, Long) =
    (b1._1 + b2._1, b1._2 + b2._2)
  override def finish(reduction: (Long, Long)): Double = reduction._1.toDouble / reduction._2
  override def bufferEncoder: Encoder[(Long, Long)] =
    Encoders.tuple(Encoders.scalaLong, Encoders.scalaLong)
  override def outputEncoder: Encoder[Double] = Encoders.scalaDouble
}

class AggregateMessagesDatasetSuite extends SparkFunSuite with GraphFrameTestSparkContext {

  import sqlImplicits.*

  test("sendToSrc and sendToDst with mergeMsg") {
    val vertices = spark.createDataset(
      Seq(
        GraphVertex(1L, UserProfile("Alice", 34)),
        GraphVertex(2L, UserProfile("Bob", 36)),
        GraphVertex(3L, UserProfile("Charlie", 30)),
        GraphVertex(4L, UserProfile("Isolated", 50))))
    val edges = spark.createDataset(
      Seq(GraphEdge(1L, 2L, "friend"), GraphEdge(2L, 3L, "follow"), GraphEdge(3L, 1L, "friend")))
    val g = GraphDataset(vertices, edges)

    // For each user, sum adjacent ages + 1 for src when relationship is "friend"
    val agg = g
      .aggregateMessages[Int]
      .sendToSrc(t => t.dstAttr.age + (if (t.attr == "friend") 1 else 0))
      .sendToDst(t => t.srcAttr.age)
      .agg(_ + _)
      .collect()
      .toMap

    // 1: dst=2 (36 + 1) from (1->2), src=3 (30) from (3->1) => 67
    // 2: dst=3 (30) from (2->3), src=1 (34) from (1->2) => 64
    // 3: dst=1 (34 + 1) from (3->1), src=2 (36) from (2->3) => 71
    // 4: isolated, receives no messages
    assert(agg === Map(1L -> 67, 2L -> 64, 3L -> 71))
  }

  test("conditional sendToSrcOpt and sendToDstOpt") {
    val vertices =
      spark.createDataset(Seq(GraphVertex(1L, 10), GraphVertex(2L, 20), GraphVertex(3L, 30)))
    val edges = spark.createDataset(
      Seq(GraphEdge(1L, 2L, "keep"), GraphEdge(2L, 3L, "drop"), GraphEdge(1L, 3L, "keep")))
    val g = GraphDataset(vertices, edges)

    val agg = g
      .aggregateMessages[Int]
      .sendToDstOpt(t => if (t.attr == "keep") Some(t.srcAttr) else None)
      .agg(_ + _)
      .collect()
      .toMap

    assert(agg === Map(2L -> 10, 3L -> 10))
  }

  test("AggregateMessagesDataset.run one-shot API") {
    val vertices =
      spark.createDataset(Seq(GraphVertex(1L, 10), GraphVertex(2L, 20), GraphVertex(3L, 30)))
    val edges = spark.createDataset(Seq(GraphEdge(1L, 2L, 1), GraphEdge(2L, 3L, 1)))
    val g = GraphDataset(vertices, edges)

    val agg = AggregateMessagesDataset
      .run[Int, Int, Int](
        graph = g,
        sendMsg = t => Iterator((t.dstId, t.srcAttr), (t.srcId, t.dstAttr)),
        mergeMsg = _ + _)
      .collect()
      .toMap

    assert(agg === Map(1L -> 20, 2L -> 40, 3L -> 20))
  }

  test("typed Aggregator with different output type") {
    val vertices =
      spark.createDataset(Seq(GraphVertex(1L, 10), GraphVertex(2L, 20), GraphVertex(3L, 40)))
    val edges = spark.createDataset(Seq(GraphEdge(1L, 3L, 1), GraphEdge(2L, 3L, 1)))
    val g = GraphDataset(vertices, edges)

    val avgIncoming = g
      .aggregateMessages[Int]
      .sendToDst(t => t.srcAttr)
      .agg(AvgIntAggregator)
      .collect()
      .toMap

    assert(avgIncoming.keySet === Set(3L))
    assert(avgIncoming(3L) === 15.0 +- 1e-6)
  }

  test("throws when no senders are configured") {
    val vertices = spark.createDataset(Seq(GraphVertex(1L, 1)))
    val edges = spark.emptyDataset[GraphEdge[Int]]
    val g = GraphDataset(vertices, edges)

    intercept[IllegalArgumentException] {
      g.aggregateMessages[Int].agg(_ + _)
    }
  }

  test("custom case class VD, ED, and message type deserialize cleanly") {
    val vertices = spark.createDataset(
      Seq(
        GraphVertex(1L, UserProfile("Alice", 30)),
        GraphVertex(2L, UserProfile("Bob", 40)),
        GraphVertex(3L, UserProfile("Charlie", 50))))
    val edges = spark.createDataset(
      Seq(GraphEdge(1L, 3L, LinkMeta("friend", 2)), GraphEdge(2L, 3L, LinkMeta("colleague", 1))))
    val g = GraphDataset(vertices, edges)

    val agg = g
      .aggregateMessages[AgeSummary]
      .sendToDst(t => AgeSummary(t.srcAttr.age + t.attr.bonus, 1))
      .agg((a, b) => AgeSummary(a.totalAge + b.totalAge, a.count + b.count))
      .collect()
      .toMap

    assert(agg === Map(3L -> AgeSummary(73, 2)))
  }
}
