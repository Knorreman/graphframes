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

## Connected Components

`GraphDataset.connectedComponents` wraps `GraphFrame.connectedComponents` (supporting `two_phase`,
`randomized_contraction`, and `graphx`) and returns a typed `Dataset[ConnectedComponentResult]`:

```scala
import org.graphframes.lib.ConnectedComponents

val cc = g.connectedComponents
  .setAlgorithm(ConnectedComponents.ALGO_TWO_PHASE)
  .setUseLocalCheckpoints(true)
  .run()
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
