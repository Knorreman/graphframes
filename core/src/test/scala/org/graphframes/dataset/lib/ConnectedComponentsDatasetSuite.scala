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

package org.graphframes.dataset.lib

import org.apache.spark.storage.StorageLevel
import org.graphframes.GraphFrameTestSparkContext
import org.graphframes.SparkFunSuite
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphEdge
import org.graphframes.dataset.GraphVertex
import org.graphframes.lib.ConnectedComponents

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

  test("respects GraphFramesConf settings and releases raw DF cache on unpersist") {
    val algoKey = "spark.graphframes.connectedComponents.algorithm"
    val localCkptKey = "spark.graphframes.useLocalCheckpoints"
    val prevAlgo = spark.conf.getOption(algoKey)
    val prevLocal = spark.conf.getOption(localCkptKey)

    spark.conf.set(algoKey, ConnectedComponents.ALGO_RANDOMIZED_CONTRACTION)
    spark.conf.set(localCkptKey, "true")
    try {
      val vv = spark.createDataset((1L to 4L).map(i => GraphVertex(i, s"v$i"))).cache()
      val ee = spark.createDataset(Seq(GraphEdge(1L, 2L, 1), GraphEdge(3L, 4L, 1))).cache()
      assert(vv.count() === 4L)
      assert(ee.count() === 2L)
      val beforeIds = sc.getPersistentRDDs.keySet.toSet

      val g = GraphDataset(vv, ee)
      val builder = g.connectedComponents
      assert(builder.getAlgorithm === ConnectedComponents.ALGO_RANDOMIZED_CONTRACTION)
      assert(builder.getUseLocalCheckpoints === true)

      val res = builder.run()
      assert(res.storageLevel !== StorageLevel.NONE)
      val groups = res.collect().groupBy(_.connectedComponent).values.map(_.map(_.id).toSet).toSet
      assert(groups === Set(Set(1L, 2L), Set(3L, 4L)))

      res.unpersist(blocking = true)
      assert(res.storageLevel === StorageLevel.NONE)
      assert(vv.unpersist(blocking = true).storageLevel === StorageLevel.NONE)
      assert(ee.unpersist(blocking = true).storageLevel === StorageLevel.NONE)
      val leaked = sc.getPersistentRDDs.filter { case (id, rdd) =>
        !beforeIds.contains(id) && !rdd.isCheckpointed
      }
      assert(leaked.isEmpty, s"Unexpected persistent RDDs: ${leaked.values.map(_.name)}")
    } finally {
      prevAlgo match {
        case Some(v) => spark.conf.set(algoKey, v)
        case None => spark.conf.unset(algoKey)
      }
      prevLocal match {
        case Some(v) => spark.conf.set(localCkptKey, v)
        case None => spark.conf.unset(localCkptKey)
      }
    }
  }
}
