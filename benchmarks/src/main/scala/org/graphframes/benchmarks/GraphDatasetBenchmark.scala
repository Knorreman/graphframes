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

package org.graphframes.benchmarks

import org.apache.spark.graphframes.graphx.EdgeDirection
import org.apache.spark.graphframes.graphx.Graph
import org.apache.spark.graphframes.graphx.Pregel as GraphXPregel
import org.apache.spark.sql.functions.*
import org.apache.spark.storage.StorageLevel
import org.graphframes.GraphDataset
import org.graphframes.lib.AggregateMessages
import org.graphframes.lib.GraphEdge
import org.graphframes.lib.GraphVertex
import org.graphframes.lib.Pregel
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

import java.util.concurrent.TimeUnit

@State(Scope.Benchmark)
@Warmup(iterations = 1)
@Measurement(iterations = 3)
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.SECONDS)
@Fork(
  value = 1,
  jvmArgs = Array(
    "-Xmx10g",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
    "--add-opens=java.base/java.util=ALL-UNNAMED"))
class GraphDatasetBenchmark extends LDBCBenchmarkBase {

  @Param(Array("graphdataset", "graphframe", "graphx"))
  var api: String = _

  var typedGraph: GraphDataset[Long, Int] = _
  var graphX: Graph[Long, Int] = _

  override def loadGraph(name: String): Unit = {
    super.loadGraph(name)
    val session = spark
    import session.implicits.*

    val vv = graph.vertices
      .select(col("id").cast("long").alias("id"), col("id").cast("long").alias("attr"))
      .as[GraphVertex[Long]]
      .persist(StorageLevel.MEMORY_AND_DISK_SER)
    val ee = graph.edges
      .select(
        col("src").cast("long").alias("srcId"),
        col("dst").cast("long").alias("dstId"),
        lit(1).alias("attr"))
      .as[GraphEdge[Int]]
      .persist(StorageLevel.MEMORY_AND_DISK_SER)

    typedGraph = GraphDataset(vv, ee)
    graphX = typedGraph.toGraphX.cache()
    val _ = (
      typedGraph.vertices.count(),
      typedGraph.edges.count(),
      graphX.vertices.count(),
      graphX.edges.count())
  }

  @Benchmark
  def benchmarkDegrees(blackhole: Blackhole): Unit = {
    api match {
      case "graphdataset" =>
        val res = typedGraph.degrees.write.format("noop").mode("overwrite").save()
        blackhole.consume(res)
      case "graphframe" =>
        val res = graph.degrees.write.format("noop").mode("overwrite").save()
        blackhole.consume(res)
      case "graphx" =>
        val res = graphX.ops.degrees.count()
        blackhole.consume(res)
    }
  }

  @Benchmark
  def benchmarkAggregateMessages(blackhole: Blackhole): Unit = {
    val session = spark
    import session.implicits.*
    api match {
      case "graphdataset" =>
        val agg = typedGraph
          .aggregateMessages[Long]
          .sendToDst(_.srcAttr)
          .agg(_ + _)
        val res = agg.write.format("noop").mode("overwrite").save()
        blackhole.consume(res)
      case "graphframe" =>
        val agg = graph.aggregateMessages
          .sendToDst(AggregateMessages.src("id").cast("long"))
          .agg(sum(AggregateMessages.msg).as("sum_id"))
        val res = agg.write.format("noop").mode("overwrite").save()
        agg.unpersist()
        blackhole.consume(res)
      case "graphx" =>
        val agg = graphX.aggregateMessages[Long](ctx => ctx.sendToDst(ctx.srcAttr), _ + _)
        val res = agg.count()
        blackhole.consume(res)
    }
  }

  @Benchmark
  def benchmarkPregel(blackhole: Blackhole): Unit = {
    val session = spark
    import session.implicits.*
    val maxIter = 3
    api match {
      case "graphdataset" =>
        val result = typedGraph.pregel
          .setMaxIter(maxIter)
          .setCheckpointInterval(2)
          .setUseLocalCheckpoints(useLocalCheckpoints.toBoolean)
          .setActiveDirection(EdgeDirection.Out)
          .withMessages[Long](Long.MinValue)
          .vprog((_, attr, msg) => math.max(attr, msg))
          .sendMsgToDstOpt(t => if (t.srcAttr > t.dstAttr) Some(t.srcAttr) else None)
          .mergeMsg((a, b) => math.max(a, b))
          .run()
        val res = result.vertices.write.format("noop").mode("overwrite").save()
        result.vertices.unpersist()
        blackhole.consume(res)
      case "graphframe" =>
        val result = graph.pregel
          .setMaxIter(maxIter)
          .setCheckpointInterval(2)
          .setUseLocalCheckpoints(useLocalCheckpoints.toBoolean)
          .setEarlyStopping(true)
          .withVertexColumn(
            "max_val",
            col("id").cast("long"),
            greatest(col("max_val"), coalesce(Pregel.msg, col("max_val"))))
          .sendMsgToDst(
            when(Pregel.src("max_val") > Pregel.dst("max_val"), Pregel.src("max_val")))
          .aggMsgs(max(Pregel.msg))
          .run()
        val res = result.write.format("noop").mode("overwrite").save()
        result.unpersist()
        blackhole.consume(res)
      case "graphx" =>
        val result = GraphXPregel(graphX, Long.MinValue, maxIter, EdgeDirection.Out)(
          vprog = (_, attr, msg) => math.max(attr, msg),
          sendMsg =
            t => if (t.srcAttr > t.dstAttr) Iterator((t.dstId, t.srcAttr)) else Iterator.empty,
          mergeMsg = math.max)
        val res = result.vertices.count()
        result.unpersist()
        blackhole.consume(res)
    }
  }
}
