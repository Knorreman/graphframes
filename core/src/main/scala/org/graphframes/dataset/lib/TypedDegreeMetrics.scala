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

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.collect_list
import org.apache.spark.sql.functions.count
import org.apache.spark.sql.functions.map_from_entries
import org.apache.spark.sql.functions.struct
import org.apache.spark.sql.functions.when
import org.graphframes.dataset.GraphVertex
import org.graphframes.dataset.GraphVertexBase
import org.graphframes.dataset.VertexId

import scala.reflect.runtime.universe.TypeTag

/** Result row for Power Iteration Clustering on [[org.graphframes.dataset.GraphDataset]]. */
case class ClusterAssignment(id: VertexId, cluster: Int) extends GraphVertexBase[Int] {
  def attr: Int = cluster
  def toVertex: GraphVertex[Int] = GraphVertex(id, cluster)
}

object ClusterAssignment {
  implicit val encoder: Encoder[ClusterAssignment] =
    Encoders.product[ClusterAssignment]
}

/**
 * Result row for typed per-edge-type degree metrics (`typeOutDegree`, `typeInDegree`,
 * `typeDegree`).
 */
case class TypedDegreeResult[K](id: VertexId, degrees: Map[K, Int])
    extends GraphVertexBase[Map[K, Int]] {
  def attr: Map[K, Int] = degrees
  def toVertex: GraphVertex[Map[K, Int]] = GraphVertex(id, degrees)
}

object TypedDegreeResult {
  implicit def encoder[K: TypeTag]: Encoder[TypedDegreeResult[K]] =
    Encoders.product[TypedDegreeResult[K]]
}

private[dataset] case class TypedDegreePair[K](id: VertexId, k: K) extends Serializable

private[dataset] object TypedDegreePair {
  implicit def encoder[K: TypeTag]: Encoder[TypedDegreePair[K]] =
    Encoders.product[TypedDegreePair[K]]
}

private[dataset] case class TypedDegreeOptPair[K](id: VertexId, k: Option[K]) extends Serializable

private[dataset] object TypedDegreeOptPair {
  implicit def encoder[K: TypeTag]: Encoder[TypedDegreeOptPair[K]] =
    Encoders.product[TypedDegreeOptPair[K]]
}

private[dataset] object TypedDegreeMetrics {
  private val ID_COL = "id"
  private val KEY_COL = "k"
  private val CNT_COL = "cnt"
  private val DEGREES_COL = "degrees"

  def compute[K: TypeTag](
      pairs: Dataset[TypedDegreePair[K]],
      edgeTypes: Option[Seq[K]]): Dataset[TypedDegreeResult[K]] = {
    edgeTypes match {
      case Some(allowed) =>
        val allowedList = allowed.filter(k => Option(k).isDefined).distinct.toVector
        val allowedSet = allowedList.toSet
        pairs
          .map { p =>
            val filteredKey = Option(p.k).filter(allowedSet.contains)
            TypedDegreeOptPair(p.id, filteredKey)
          }
          .toDF(ID_COL, KEY_COL)
          .groupBy(col(ID_COL), col(KEY_COL))
          .agg(count("*").cast("int").alias(CNT_COL))
          .groupBy(col(ID_COL))
          .agg(map_from_entries(
            collect_list(when(col(KEY_COL).isNotNull, struct(col(KEY_COL), col(CNT_COL)))))
            .alias(DEGREES_COL))
          .as[TypedDegreeResult[K]]
          .map { r =>
            val existing = Option(r.degrees).getOrElse(Map.empty[K, Int])
            TypedDegreeResult(r.id, allowedList.map(k => k -> existing.getOrElse(k, 0)).toMap)
          }
      case None =>
        pairs
          .toDF(ID_COL, KEY_COL)
          .filter(col(KEY_COL).isNotNull)
          .groupBy(col(ID_COL), col(KEY_COL))
          .agg(count("*").cast("int").alias(CNT_COL))
          .groupBy(col(ID_COL))
          .agg(map_from_entries(collect_list(struct(col(KEY_COL), col(CNT_COL))))
            .alias(DEGREES_COL))
          .as[TypedDegreeResult[K]]
    }
  }
}
