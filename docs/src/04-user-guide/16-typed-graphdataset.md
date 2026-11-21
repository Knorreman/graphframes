# Typed GraphDataset API (Scala)

`GraphDataset[VD, ED]` is a strongly typed Scala API backed by Spark `Dataset`s of
`GraphVertex[VD]` and `GraphEdge[ED]`. It combines compile-time type safety for vertex and edge
attributes with Spark SQL's Catalyst optimizer and full interoperability with `GraphFrame` and
GraphX.

For full API details, refer to the @:scaladoc(org.graphframes.GraphDataset).

## Creating a GraphDataset

Vertices are represented as `GraphVertex[VD](id: VertexId, attr: VD)` and edges as
`GraphEdge[ED](srcId: VertexId, dstId: VertexId, attr: ED)` where `VertexId = Long`.
Attributes can be primitive types or custom Scala case classes:

```scala
import org.graphframes.GraphDataset
import org.graphframes.lib.{GraphEdge, GraphVertex}
import spark.implicits._

case class Person(name: String, age: Int)
case class Relation(kind: String, weight: Double)

val vertices = spark.createDataset(Seq(
  GraphVertex(1L, Person("Alice", 34)),
  GraphVertex(2L, Person("Bob", 36)),
  GraphVertex(3L, Person("Charlie", 30))
))

val edges = spark.createDataset(Seq(
  GraphEdge(1L, 2L, Relation("friend", 1.0)),
  GraphEdge(2L, 3L, Relation("follow", 0.5)),
  GraphEdge(3L, 1L, Relation("friend", 0.8))
))

val g: GraphDataset[Person, Relation] = GraphDataset(vertices, edges)
```

You can also construct a `GraphDataset` directly from edges (inferring distinct vertex IDs with a
default vertex attribute):

```scala
val gFromEdges = GraphDataset.fromEdges(edges, defaultVertexAttribute = Person("Unknown", 0))
```

### Converting to and from GraphFrame / GraphX

```scala
// GraphDataset <-> GraphFrame
val gf = g.toGraphFrame
val g2 = GraphDataset.fromGraphFrame[Person, Relation](gf)

// GraphDataset -> GraphX Graph[VD, ED]
val gx = g.toGraphX
```

## Basic Operations, Subgraphs, and Transformations

```scala
// Degree metrics: Dataset[(VertexId, Int)]
val inDeg = g.inDegrees
val outDeg = g.outDegrees
val deg = g.degrees

// Typed triplets: Dataset[GraphTriplet[Person, Relation]]
val triplets = g.triplets

// Typed filtering and subgraph selection
val olderOnly = g.filterVertices(v => v.attr.age >= 32)
val friendsOnly = g.filterEdges(e => e.attr.kind == "friend")
val sub = g.subgraph(vpred = _.attr.age >= 32, epred = _.attr.kind == "friend")
val connectedOnly = g.dropIsolatedVertices()

// Typed attribute transformations
val ages = g.mapVertices((_, person) => person.age)
val weights = g.mapEdges(e => e.attr.weight)
val descriptions = g.mapTriplets(t => s"${t.srcAttr.name} -> ${t.dstAttr.name}")

// Graph utilities
val reversed = g.asReversed()
val undirected = g.asUndirected()
g.validate()
```

## Typed AggregateMessages

`GraphDataset.aggregateMessages[M]` provides a compile-time type-safe message-passing primitive
over `GraphTriplet[VD, ED]`.

For API details, refer to the @:scaladoc(org.graphframes.lib.AggregateMessagesDataset).

```scala
// Builder API with sendToSrc / sendToDst (and optional sendToSrcOpt / sendToDstOpt)
val summedAges = g
  .aggregateMessages[Int]
  .sendToSrc(t => t.dstAttr.age)
  .sendToDst(t => t.srcAttr.age)
  .agg(_ + _)

// Conditional messages
val friendAges = g
  .aggregateMessages[Int]
  .sendToDstOpt(t => if (t.attr.kind == "friend") Some(t.srcAttr.age) else None)
  .agg(_ + _)
```

You can also pass a typed Spark `Aggregator[M, _, R]` to `agg(...)` when the output type `R`
differs from the message type `M` (e.g., computing an average), or use the one-shot helper
`AggregateMessagesDataset.run(graph, sendMsg, mergeMsg)`.

## Typed Pregel

`GraphDataset.pregel` implements a typed bulk-synchronous message-passing API supporting
`EdgeDirection` (`Out`, `In`, `Both`, `Either`), early termination when no messages remain, and
both reliable and local checkpointing.

For API details, refer to the @:scaladoc(org.graphframes.lib.PregelGraphDataset).

### Fluent Typed Builder

```scala
import org.apache.spark.graphframes.graphx.EdgeDirection

val result = g
  .mapVertices((id, _) => if (id == 1L) 1 else 0)
  .pregel
  .setMaxIter(10)
  .setCheckpointInterval(2)
  .setUseLocalCheckpoints(true)
  .setActiveDirection(EdgeDirection.Out)
  .withMessages[Int](0)
  .vprog((_, attr, msg) => math.max(attr, msg))
  .sendMsgToDstOpt(t => if (t.srcAttr > t.dstAttr) Some(t.srcAttr) else None)
  .mergeMsg(math.max)
  .run()
```

### Functional `runTriplets` / GraphX-Compatible `run`

```scala
// Using GraphTriplet[VD, ED]
val shortest = weightedGraph.pregel
  .setMaxIter(10)
  .setActiveDirection(EdgeDirection.Out)
  .runTriplets[Int](Int.MaxValue)(
    vprog = (_, dist, msg) => math.min(dist, msg),
    sendMsg = t =>
      if (t.srcAttr != Int.MaxValue && t.srcAttr + t.attr < t.dstAttr)
        Iterator((t.dstId, t.srcAttr + t.attr))
      else Iterator.empty,
    mergeMsg = math.min
  )
```

## Typed Traversals (`bfs`, `allPaths`, `aggregateNeighbors`)

```scala
// Breadth-First Search -> Dataset[GraphPath[VD, ED]]
val shortestPaths = g.bfs
  .fromPred(_.attr.name == "Alice")
  .toPred(_.attr.name == "Charlie")
  .edgeFilter(_.attr.kind == "friend")
  .maxPathLength(5)
  .run()

// All Simple Paths -> Dataset[SimplePath]
val allSimple = g.allPaths
  .fromPred(_.attr.name == "Alice")
  .toPred(_.attr.name == "Charlie")
  .maxPathLength(4)
  .setUseLocalCheckpoints(true)
  .run()

// Multi-Hop Neighbor Aggregation -> Dataset[NeighborAggregateResult[S]]
val traversed = g.aggregateNeighbors
  .setStartingVertices(_.attr.name == "Alice")
  .setMaxHops(3)
  .setUseLocalCheckpoints(true)
  .withAccumulator[Double](
    init = _ => 0.0,
    update = (weightSum, triplet) => weightSum + triplet.attr.weight)
  .setTargetCondition((_, triplet) => triplet.dstAttr.name == "Charlie")
  .run()
```

## Standard Graph Algorithms

`GraphDataset` wraps the standard `GraphFrame` topology algorithms with strongly typed results:

```scala
import org.graphframes.lib.ConnectedComponents

// Connected Components -> Dataset[ConnectedComponentResult]
val cc = g.connectedComponents
  .setAlgorithm(ConnectedComponents.ALGO_TWO_PHASE)
  .setUseLocalCheckpoints(true)
  .run()

// PageRank -> GraphDataset[PageRankAttr[VD], WeightedEdgeAttr[ED]]
val pr = g.pageRank.resetProbability(0.15).maxIter(5).run()

// Shortest Paths -> Dataset[ShortestPathsResult]
val sp = g.shortestPaths.landmarks(Seq(1L, 3L)).run()

// Triangle Count -> Dataset[TriangleCountResult]
val tc = g.triangleCount.run()

// Label Propagation -> Dataset[LabelPropagationResult]
val lp = g.labelPropagation.maxIter(5).run()

// Strongly Connected Components -> Dataset[ConnectedComponentResult]
val scc = g.stronglyConnectedComponents.maxIter(5).run()

// K-Core Decomposition -> Dataset[KCoreResult]
val kc = g.kCore.setUseLocalCheckpoints(true).run()

// Maximal Independent Set -> Dataset[GraphVertex[VD]]
val mis = g.maximalIndependentSet.setUseLocalCheckpoints(true).run(seed = 42L)

// Cycle Detection -> Dataset[Seq[VertexId]]
val cycles = g.detectingCycles.setUseLocalCheckpoints(true).run()

// Parallel Personalized PageRank -> GraphDataset[PersonalizedPageRankAttr[VD], WeightedEdgeAttr[ED]]
val ppr = g.parallelPersonalizedPageRank.sourceIds(Seq(1L, 2L)).maxIter(5).run()

// Structure-Aware Label Propagation (Spark 4.1+) -> Dataset[LabelPropagationResult]
// or attribute-guided via runWithInitialLabel(_.attr) -> Dataset[GraphVertex[L]]
val salp = g.structureAwareLabelPropagation.maxIter(5).setUseLocalCheckpoints(true).run()

// HyperANF -> Dataset[HyperANFResult]
val anf = g.hyperANF.setNHops(3).run()

// Random-Walk Node Embeddings -> Dataset[NodeEmbeddingResult]
val emb = g.randomWalksBasedEmbedding.setRandomWalks(rw).setSequenceModel(Right(h2v)).run()

// SVD++ with typed edge rating extractor -> (Dataset[SVDPlusPlusResult], Double)
val (svdVertices, loss) = g.svdPlusPlus.rank(8).maxIter(5).run(_.weight)

// Power Iteration Clustering with optional typed weight extractor -> Dataset[ClusterAssignment]
val clusters = g.powerIterationClustering(k = 2, maxIter = 20, weightExtractor = Some(_.weight))

// Per-edge-type degree metrics -> Dataset[TypedDegreeResult[K]]
val outByKind = g.typeOutDegree(_.kind)
val inByKind = g.typeInDegree(_.kind, edgeTypes = Some(Seq("friend", "follow")))
val degByKind = g.typeDegree(_.kind)
```

## Saving and Loading

Because `vertices` and `edges` are standard Spark `Dataset`s, they can be saved and loaded with
any Spark data source such as Parquet:

```scala
g.vertices.write.parquet("/path/to/vertices")
g.edges.write.parquet("/path/to/edges")

val loaded = GraphDataset(
  spark.read.parquet("/path/to/vertices").as[GraphVertex[Person]],
  spark.read.parquet("/path/to/edges").as[GraphEdge[Relation]]
)
```

## Java API (`JavaGraphDataset`)

For Java applications, `JavaGraphDataset<VD, ED>` provides a native Java-friendly typed API that
uses explicit Spark `Encoder`s (`Encoders.STRING()`, `Encoders.INT()`, `Encoders.bean(...)`, etc.)
and Spark Java functional interfaces (`FilterFunction`, `MapFunction`, `Function2`, `Function3`,
`ReduceFunction`, `Optional`) without requiring Scala `TypeTag`s or implicits:

```java
import org.apache.spark.graphframes.graphx.EdgeDirection;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.graphframes.JavaGraphDataset;
import scala.Tuple2;
import scala.Tuple3;
import java.util.Optional;

JavaGraphDataset<Person, Relation> g = JavaGraphDataset.of(
    verticesDs, // Dataset<Tuple2<Long, Person>>
    edgesDs,    // Dataset<Tuple3<Long, Long, Relation>>
    Encoders.bean(Person.class),
    Encoders.bean(Relation.class));

// Typed message aggregation
Dataset<Tuple2<Long, Integer>> ageSum = g
    .aggregateMessages(Encoders.INT())
    .sendToDst(t -> t.srcAttr().getAge())
    .agg(Integer::sum);

// Typed Pregel
JavaGraphDataset<Integer, Relation> maxAges = g
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
```

See `org.graphframes.examples.GraphDatasetExample` for a complete runnable Java example.
