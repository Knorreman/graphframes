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

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.storage.StorageLevel
import org.graphframes.GraphFrame
import org.graphframes.dataset.GraphDataset
import org.graphframes.dataset.GraphEdge
import org.graphframes.dataset.GraphVertex

import scala.reflect.runtime.universe.TypeTag

private[dataset] trait DatasetArguments {
  private[dataset] def check[A](a: Option[A], name: String): A = {
    a.getOrElse(throw new IllegalArgumentException(s"Param $name is required."))
  }
}

private[dataset] object DatasetCacheUtils {

  /**
   * Persists and materializes a typed `Dataset` projected from a wrapped `GraphFrame` algorithm
   * result `DataFrame`, then unpersists `raw` so the intermediate DataFrame does not leak in
   * Spark's cache manager.
   */
  def materializeAndReleaseRaw[T](
      raw: DataFrame,
      typed: Dataset[T],
      storageLevel: StorageLevel): Dataset[T] = {
    if (typed.storageLevel != StorageLevel.NONE) {
      val _ = typed.count()
      typed
    } else {
      val persisted = typed.persist(storageLevel)
      var succeeded = false
      try {
        val _ = persisted.count()
        succeeded = true
        persisted
      } finally {
        if (!succeeded) {
          val _ = persisted.unpersist()
        }
        if (raw.storageLevel != StorageLevel.NONE) {
          val _ = raw.unpersist()
        }
        ()
      }
    }
  }

  /**
   * Persists and materializes both vertex and edge datasets projected from a wrapped `GraphFrame`
   * algorithm result, then unpersists `rawGF` so neither raw DataFrame leaks in Spark's cache
   * manager.
   */
  def materializeGraphAndReleaseRaw[VD: TypeTag, ED: TypeTag](
      rawGF: GraphFrame,
      typedVertices: Dataset[GraphVertex[VD]],
      typedEdges: Dataset[GraphEdge[ED]],
      storageLevel: StorageLevel): GraphDataset[VD, ED] = {
    val verticesAlreadyCached = typedVertices.storageLevel != StorageLevel.NONE
    val edgesAlreadyCached = typedEdges.storageLevel != StorageLevel.NONE
    val persistedV =
      if (verticesAlreadyCached) typedVertices else typedVertices.persist(storageLevel)
    val persistedE =
      if (edgesAlreadyCached) typedEdges else typedEdges.persist(storageLevel)
    var succeeded = false
    try {
      val _ = (persistedV.count(), persistedE.count())
      succeeded = true
      GraphDataset(persistedV, persistedE)
    } finally {
      if (!succeeded) {
        if (!verticesAlreadyCached) {
          val _ = persistedV.unpersist()
        }
        if (!edgesAlreadyCached) {
          val _ = persistedE.unpersist()
        }
      }
      if (!verticesAlreadyCached && rawGF.vertices.storageLevel != StorageLevel.NONE) {
        val _ = rawGF.vertices.unpersist()
      }
      if (!edgesAlreadyCached && rawGF.edges.storageLevel != StorageLevel.NONE) {
        val _ = rawGF.edges.unpersist()
      }
      ()
    }
  }
}
