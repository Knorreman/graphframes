package org.graphframes

import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.{Dataset, SparkSession}
import org.graphframes.lib.{GraphEdge, GraphVertex}

import java.io.File
import java.nio.file.Files

class GraphDatasetSuite extends SparkFunSuite with GraphFrameTestSparkContext {
    val localVertices: Map[Long, String] = Map(1L -> "A", 2L -> "B", 3L -> "C")
    val localEdges: Map[(Long, Long), String] = Map((1L, 2L) -> "love", (2L, 1L) -> "hate", (2L, 3L) -> "follow")
    var edges: Dataset[GraphEdge[String]] = _
    var vertices: Dataset[GraphVertex[String]] = _
    var tempDir: File = _
    implicit lazy val _spark: SparkSession = sparkSession

    override def beforeAll(): Unit = {
      super.beforeAll()
      import sqlImplicits._
      tempDir = Files.createTempDirectory(this.getClass.getName).toFile
      vertices = spark.createDataset(localVertices.toSeq.map(tpl => GraphVertex(tpl._1, tpl._2)))
      edges = spark.createDataset(localEdges.toSeq.map {
          case ((src, dst), action) =>
              GraphEdge(src, dst, action)
          })
    }

    override def afterAll(): Unit = {
        FileUtils.deleteQuietly(tempDir)
        super.afterAll()
    }

    
    test("save/load") {
      import sqlImplicits._
      val g0 = GraphDataset(vertices, edges)
      val vPath = new Path(tempDir.getPath, "vertices").toString
      val ePath = new Path(tempDir.getPath, "edges").toString
      g0.vertices.write.parquet(vPath)
      g0.edges.write.parquet(ePath)

      val v1 = spark.read.parquet(vPath).as[GraphVertex[String]]
      val e1 = spark.read.parquet(ePath).as[GraphEdge[String]]
      val g1 = GraphDataset(v1, e1)

      g1.vertices.collect().foreach { case GraphVertex(id: Long, name: String) =>
      assert(localVertices(id) === name)
      }
      g1.edges.collect().foreach { case GraphEdge(src: Long, dst: Long, action: String) =>
      assert(localEdges((src, dst)) === action)
      }

    }

    test("degree metrics") {
      import sqlImplicits._
      val g: GraphDataset[String, String] = GraphDataset(vertices, edges)

      val outDegrees = g.outDegrees.collect()
          .map { case (id: Long, outDeg: Int) =>
              (id, outDeg)
          }.toMap
      assert(outDegrees === Map(1L -> 1, 2L -> 2))

      val inDegrees = g.inDegrees.collect().map { case (id: Long, inDeg: Int) =>
          (id, inDeg)
      }.toMap
      assert(inDegrees === Map(1L -> 1, 2L -> 1, 3L -> 1))

      val degrees = g.degrees.collect().map { case (id: Long, deg: Int) =>
          (id, deg)
      }.toMap
      assert(degrees === Map(1L -> 2, 2L -> 3, 3L -> 1))
    }

}
