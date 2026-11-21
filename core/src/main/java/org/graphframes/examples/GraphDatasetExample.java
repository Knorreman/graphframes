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

package org.graphframes.examples;

import org.apache.spark.graphframes.graphx.EdgeDirection;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SparkSession;
import org.graphframes.dataset.JavaGraphDataset;
import org.graphframes.dataset.lib.ConnectedComponentResult;
import scala.Tuple2;
import scala.Tuple3;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Optional;

/**
 * Example demonstrating the typed Java API ({@link org.graphframes.dataset.JavaGraphDataset}) with JavaBean vertex/edge
 * attributes, {@code aggregateMessages}, {@code pregel}, and {@code connectedComponents}.
 */
public class GraphDatasetExample {

  public static class Person implements Serializable {
    private String name;
    private int age;

    public Person() {}

    public Person(String name, int age) {
      this.name = name;
      this.age = age;
    }

    public String getName() {
      return name;
    }

    public void setName(String name) {
      this.name = name;
    }

    public int getAge() {
      return age;
    }

    public void setAge(int age) {
      this.age = age;
    }
  }

  public static void main(String[] args) {
    SparkSession spark = SparkSession.builder()
        .master("local[2]")
        .appName("JavaGraphDatasetExample")
        .getOrCreate();

    try {
      Dataset<Tuple2<Long, Person>> vertices = spark.createDataset(
          Arrays.asList(
              new Tuple2<>(1L, new Person("Alice", 34)),
              new Tuple2<>(2L, new Person("Bob", 36)),
              new Tuple2<>(3L, new Person("Charlie", 30))),
          Encoders.tuple(Encoders.LONG(), Encoders.bean(Person.class)));

      Dataset<Tuple3<Long, Long, String>> edges = spark.createDataset(
          Arrays.asList(
              new Tuple3<>(1L, 2L, "friend"),
              new Tuple3<>(2L, 3L, "follow")),
          Encoders.tuple(Encoders.LONG(), Encoders.LONG(), Encoders.STRING()));

      JavaGraphDataset<Person, String> graph = JavaGraphDataset.of(
          vertices, edges, Encoders.bean(Person.class), Encoders.STRING());

      // Typed AggregateMessages: sum ages of incoming neighbors
      Dataset<Tuple2<Long, Integer>> incomingAgeSum = graph
          .aggregateMessages(Encoders.INT())
          .sendToDst(t -> t.srcAttr().getAge())
          .agg(Integer::sum);
      incomingAgeSum.show();

      // Typed Pregel: propagate maximum age along outgoing edges
      JavaGraphDataset<Integer, String> maxAges = graph
          .mapVertices((id, p) -> p.getAge(), Encoders.INT())
          .pregel()
          .setMaxIter(5)
          .setUseLocalCheckpoints(true)
          .setActiveDirection(EdgeDirection.Out())
          .withMessages(0, Encoders.INT())
          .vprog((id, age, msg) -> Math.max(age, msg))
          .sendMsgToDstOpt(t -> t.srcAttr() > t.dstAttr()
              ? Optional.of(t.srcAttr())
              : Optional.empty())
          .mergeMsg(Math::max)
          .run();
      maxAges.vertices().show();
      maxAges.vertices().unpersist();

      // Connected Components
      Dataset<ConnectedComponentResult> cc = graph.connectedComponents()
          .setUseLocalCheckpoints(true)
          .run();
      cc.show();
      cc.unpersist();
    } finally {
      spark.stop();
    }
  }
}
