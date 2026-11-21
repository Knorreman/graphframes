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

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Row;
import org.apache.spark.storage.StorageLevel;
import org.graphframes.GraphFrame;
import org.graphframes.dataset.JavaGraphDataset;
import scala.Tuple2;
import scala.Tuple3;

final class JavaDatasetCacheUtils {

  private JavaDatasetCacheUtils() {}

  static <T> Dataset<T> materializeAndReleaseRaw(
      Dataset<Row> raw, Dataset<T> typed, StorageLevel storageLevel) {
    if (!StorageLevel.NONE().equals(typed.storageLevel())) {
      typed.count();
      return typed;
    }
    Dataset<T> persisted = typed.persist(storageLevel);
    boolean succeeded = false;
    try {
      persisted.count();
      succeeded = true;
      return persisted;
    } finally {
      if (!succeeded) {
        persisted.unpersist();
      }
      if (!StorageLevel.NONE().equals(raw.storageLevel())) {
        raw.unpersist();
      }
    }
  }

  static <VD, ED> JavaGraphDataset<VD, ED> materializeGraphAndReleaseRaw(
      GraphFrame rawGF,
      Dataset<Tuple2<Long, VD>> typedVertices,
      Dataset<Tuple3<Long, Long, ED>> typedEdges,
      Encoder<VD> vdEncoder,
      Encoder<ED> edEncoder,
      StorageLevel storageLevel) {
    boolean verticesAlreadyCached = !StorageLevel.NONE().equals(typedVertices.storageLevel());
    boolean edgesAlreadyCached = !StorageLevel.NONE().equals(typedEdges.storageLevel());
    Dataset<Tuple2<Long, VD>> persistedV =
        verticesAlreadyCached ? typedVertices : typedVertices.persist(storageLevel);
    Dataset<Tuple3<Long, Long, ED>> persistedE =
        edgesAlreadyCached ? typedEdges : typedEdges.persist(storageLevel);
    boolean succeeded = false;
    try {
      persistedV.count();
      persistedE.count();
      succeeded = true;
      return JavaGraphDataset.of(persistedV, persistedE, vdEncoder, edEncoder);
    } finally {
      if (!succeeded) {
        if (!verticesAlreadyCached) {
          persistedV.unpersist();
        }
        if (!edgesAlreadyCached) {
          persistedE.unpersist();
        }
      }
      if (!verticesAlreadyCached && !StorageLevel.NONE().equals(rawGF.vertices().storageLevel())) {
        rawGF.vertices().unpersist();
      }
      if (!edgesAlreadyCached && !StorageLevel.NONE().equals(rawGF.edges().storageLevel())) {
        rawGF.edges().unpersist();
      }
    }
  }
}
