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

import org.graphframes.GraphDataset
import org.graphframes.GraphFrameTestSparkContext
import org.graphframes.SparkFunSuite

class ConnectedComponentsDatasetSuite extends SparkFunSuite with GraphFrameTestSparkContext {

  import sqlImplicits.*

  test("disconnected components with isolated vertex") {
    val vv = spark.createDataset(
      Seq(
        GraphVertex(1L, "a"),
        GraphVertex(2L, "b"),
        GraphVertex(3L, "c"),
        GraphVertex(4L, "d"),
        GraphVertex(5L, "e"),
        GraphVertex(6L, "isolated")))
    val ee = spark.createDataset(
      Seq(GraphEdge(1L, 2L, "e1"), GraphEdge(2L, 3L, "e2"), GraphEdge(4L, 5L, "e3")))
    val g = GraphDataset(vv, ee)

    Seq(1000000, 1, -1).foreach { threshold =>
      val cc = g.connectedComponents
        .setBroadcastThreshold(threshold)
        .run()
        .collect()
        .map(r => r.id -> r.connectedComponent)
        .toMap

      assert(cc === Map(1L -> 1L, 2L -> 1L, 3L -> 1L, 4L -> 4L, 5L -> 4L, 6L -> 6L))
    }
  }

  test("chain graph with local and regular checkpoints") {
    val n = 6L
    val vv = spark.createDataset((1L to n).map(i => GraphVertex(i, s"v$i")))
    val ee = spark.createDataset((1L until n).map(i => GraphEdge(i, i + 1L, 1)))
    val g = GraphDataset(vv, ee)

    Seq(true, false).foreach { useLocal =>
      val cc = g.connectedComponents
        .setCheckpointInterval(1)
        .setUseLocalCheckpoints(useLocal)
        .run()
        .collect()
        .map(r => r.id -> r.connectedComponent)
        .toMap

      assert(cc === (1L to n).map(i => i -> 1L).toMap)
    }
  }

  test("self-loops, duplicate edges, and negative vertex IDs") {
    val vv =
      spark.createDataset(Seq(GraphVertex(-10L, "a"), GraphVertex(0L, "b"), GraphVertex(5L, "c")))
    val ee = spark.createDataset(
      Seq(
        GraphEdge(-10L, -10L, "self"),
        GraphEdge(5L, -10L, "e1"),
        GraphEdge(-10L, 5L, "e2"),
        GraphEdge(5L, -10L, "dup")))
    val g = GraphDataset(vv, ee)

    val results = g.connectedComponents.run().collect()
    val cc = results.map(r => r.id -> r.connectedComponent).toMap
    assert(cc === Map(-10L -> -10L, 0L -> 0L, 5L -> -10L))

    val r0 = ConnectedComponentResult(5L, -10L)
    assert(r0.attr === -10L)
    assert(r0.toVertex === GraphVertex(5L, -10L))
  }

  test("supports all GraphFrame connectedComponents algorithms") {
    val vv = spark.createDataset(
      Seq(GraphVertex(1L, "a"), GraphVertex(2L, "b"), GraphVertex(3L, "c"), GraphVertex(4L, "d")))
    val ee = spark.createDataset(Seq(GraphEdge(1L, 2L, "e1"), GraphEdge(2L, 3L, "e2")))
    val g = GraphDataset(vv, ee)
    val expectedGroups = Set(Set(1L, 2L, 3L), Set(4L))

    Seq(
      ConnectedComponents.ALGO_TWO_PHASE,
      ConnectedComponents.ALGO_RANDOMIZED_CONTRACTION,
      ConnectedComponents.ALGO_GRAPHX).foreach { algo =>
      val cc = g.connectedComponents
        .setAlgorithm(algo)
        .run()
        .collect()
      val groups = cc.groupBy(_.connectedComponent).values.map(_.map(_.id).toSet).toSet
      assert(groups === expectedGroups, s"Algorithm $algo produced unexpected groups: $groups")
    }
  }

  test("empty edges") {
    val vv = spark.createDataset(Seq(GraphVertex(1L, "a"), GraphVertex(2L, "b")))
    val ee = spark.emptyDataset[GraphEdge[String]]
    val g = GraphDataset(vv, ee)

    val cc = g.connectedComponents.run().collect().map(r => r.id -> r.connectedComponent).toMap
    assert(cc === Map(1L -> 1L, 2L -> 2L))
  }
}
