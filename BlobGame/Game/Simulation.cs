using BlobGame.Game.GameObjects;
using BlobGame.Game.Util;
using nkast.Aether.Physics2D.Common;
using nkast.Aether.Physics2D.Dynamics;
using nkast.Aether.Physics2D.Dynamics.Contacts;
using Raylib_CsLo;

namespace BlobGame.Game;
public interface ISimulation {
    IReadOnlyGameObjectsCollection GameObjects { get; }
    eBlobType CurrentBlob { get; }
    eBlobType NextBlob { get; }
    bool CanSpawnBlob { get; }
    int Score { get; }
}

internal sealed class Simulation : ISimulation {
    internal const float GRAVITY = 111.3f;
    internal const float ARENA_WIDTH = 670;
    internal const float ARENA_HEIGHT = 846;
    internal const float ARENA_HEIGHT_LOWER = 750;
    internal const float ARENA_WALL_THICKNESS = 20;
    internal const float ARENA_SPAWN_Y_OFFSET = -22.5f;
    internal const int HIGHEST_SPAWNABLE_FRUIT_INDEX = 4;

    private GameObjectCollection _GameObjects { get; }
    public IReadOnlyGameObjectsCollection GameObjects => _GameObjects;

    private Wall GroundWall { get; set; }

    private Random Random { get; }
    private World World { get; }
    public eBlobType CurrentBlob { get; private set; }
    public eBlobType NextBlob { get; private set; }
    public bool CanSpawnBlob { get; private set; }

    public int Score { get; private set; }
    public bool IsGameOver { get; private set; }

    private List<(Guid id1, Guid id2)> HandledCollisionPairs { get; }
    private List<(eBlobType type, Vector2 position)> ToBeAdded { get; }
    private List<Blob> ToBeRemoved { get; }
    private Blob? LastSpawned { get; set; }

    internal Simulation(int seed) {
        _GameObjects = new GameObjectCollection();

        Random = new Random(seed);
        World = new World(new Vector2(0, GRAVITY));

        HandledCollisionPairs = new List<(Guid, Guid)>();
        ToBeAdded = new List<(eBlobType, Vector2 position)>();
        ToBeRemoved = new List<Blob>();
        LastSpawned = null;

        Score = 0;
        IsGameOver = false;
    }

    internal void Load() {
        CurrentBlob = GenerateRandomBlobType();
        NextBlob = GenerateRandomBlobType();
        _GameObjects.AddRange(CreateArena(World));
        GroundWall = (GameObjects.FindByName("Ground") as Wall)!;

        CanSpawnBlob = true;
    }

    internal void Update(float dT) {
        if (IsGameOver)
            return;

        for (int i = 0; i < 50; i++) {
            World.Step(dT / 50f);


            HandledCollisionPairs.Clear();
            foreach (Blob blob in ToBeRemoved)
                RemoveBlob(blob);
            ToBeRemoved.Clear();

            foreach ((eBlobType type, Vector2 position) fruit in ToBeAdded)
                CreateFruit(fruit.position, fruit.type);
            ToBeAdded.Clear();
        }

        CheckGameOver();
    }

    internal bool TrySpawnBlob(float t, out Blob? fruit) {
        fruit = null;
        if (!CanSpawnBlob || IsGameOver)
            return false;

        eBlobType type = CurrentBlob;

        CurrentBlob = NextBlob;
        NextBlob = GenerateRandomBlobType();

        float x = (t - 0.5f) * ARENA_WIDTH;
        float y = ARENA_SPAWN_Y_OFFSET;

        fruit = CreateFruit(new Vector2(x, y), type);
        LastSpawned = fruit;
        CanSpawnBlob = false;
        return true;
    }

    private Blob CreateFruit(Vector2 position, eBlobType type) {
        Blob fruit = Blob.Create(World, position, type);
        _GameObjects.Add(fruit);
        fruit.Body.OnCollision += OnBlobCollision;

        return fruit;
    }

    private void RemoveBlob(Blob fruit) {
        if (!GameObjects.Contains(fruit))
            return;

        _GameObjects.Remove(fruit);
        fruit.Body.OnCollision -= OnBlobCollision;
        World.Remove(fruit.Body);
    }

    private void CheckGameOver() {
        GameObjects.Enumerate(gO => {
            bool tmp = gO is Blob fruit && fruit.Position.Y <= 0 && fruit != LastSpawned;
            IsGameOver |= tmp;
        });
    }

    private bool OnBlobCollision(Fixture sender, Fixture other, Contact contact) {
        if (IsGameOver)
            return true;

        CheckForBlobSpawnReenable(sender, other);

        Blob? senderBlob = sender.Body.Tag as Blob;
        Blob? otherBlob = other.Body.Tag as Blob;

        if (senderBlob == null || otherBlob == null)
            return true;

        if (senderBlob.Type != otherBlob.Type || senderBlob.Type == eBlobType.Watermelon)
            return true;

        if (HandledCollisionPairs.Contains((otherBlob.Id, senderBlob.Id)))
            return true;

        Score += senderBlob.Score;
        Vector2 midPoint = (senderBlob.Position + otherBlob.Position) / 2f;
        ToBeRemoved.Add(senderBlob);
        ToBeRemoved.Add(otherBlob);
        ToBeAdded.Add((senderBlob.Type + 1, midPoint));

        HandledCollisionPairs.Add((senderBlob.Id, otherBlob.Id));

        return true;
    }

    private void CheckForBlobSpawnReenable(Fixture sender, Fixture other) {
        bool isLastSpawnedBlob =
            sender.Body.Tag is Blob senderBlob && senderBlob == LastSpawned ||
            other.Body.Tag is Blob otherBlob && otherBlob == LastSpawned;
        bool isGroundWall =
            sender.Body.Tag is Wall senderWall && senderWall == GroundWall ||
            other.Body.Tag is Wall otherWall && otherWall == GroundWall;
        bool isOtherBlob =
            sender.Body.Tag is Blob senderBlob2 && senderBlob2 != LastSpawned ||
            other.Body.Tag is Blob otherBlob2 && otherBlob2 != LastSpawned;

        if (isLastSpawnedBlob && (isGroundWall || isOtherBlob)) {
            LastSpawned = null;
            CanSpawnBlob = true;
        }
    }

    private eBlobType GenerateRandomBlobType() {
        return (eBlobType)Random.Next(HIGHEST_SPAWNABLE_FRUIT_INDEX + 1);
    }

    private static IEnumerable<GameObject> CreateArena(World world) {
        float totalWidth = ARENA_WIDTH + 2 * ARENA_WALL_THICKNESS;

        float x = -totalWidth / 2;
        float y = 0;

        GameObject[] walls = new GameObject[] {
            new Wall("Ground", world, new Rectangle(x, y + ARENA_HEIGHT, totalWidth, ARENA_WALL_THICKNESS)),
            new Wall("Left Wall", world, new Rectangle(x, y - 100, ARENA_WALL_THICKNESS, ARENA_HEIGHT + 100)),
            new Wall("Right Wall", world, new Rectangle(x + ARENA_WIDTH + ARENA_WALL_THICKNESS, y - 100, ARENA_WALL_THICKNESS, ARENA_HEIGHT + 100))
        };

        return walls;
    }
}
