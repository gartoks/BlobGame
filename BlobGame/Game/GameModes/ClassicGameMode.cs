using BlobGame.Game.Blobs;
using BlobGame.Game.GameObjects;
using BlobGame.Game.Util;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using nkast.Aether.Physics2D.Common;
using nkast.Aether.Physics2D.Dynamics;
using nkast.Aether.Physics2D.Dynamics.Contacts;
using Raylib_CsLo;
using System.Globalization;

namespace BlobGame.Game.GameModes;
/// <summary>
/// The game's simulation. Handles the game's logic without any drawing logic.
/// It is separated from the drawing logic to allow for a headless version of the game. *wink wink* Tutel *wink wink*
/// </summary>
internal sealed class ClassicGameMode : IGameMode {
    /// <summary>
    /// The id of the game mode. This must never change for a given game mode.
    /// </summary>
    public Guid Id { get; } = Guid.Parse("9445DAB4-AC62-40EE-B908-C89FBDE2D42C");

    /// <summary>
    /// The blob data loaded from the resource file.
    /// </summary>
    public IReadOnlyDictionary<int, BlobData> Blobs { get; private set; }

    /// <summary>
    /// Collection of all game objects. This includes blobs and walls.
    /// </summary>
    private GameObjectCollection _GameObjects { get; }
    /// <summary>
    /// Public read-only accesor for the game objects collection.
    /// </summary>
    public IReadOnlyGameObjectsCollection GameObjects => _GameObjects;

    /// <summary>
    /// The wall game object representing the arena ground.
    /// </summary>
    private Wall GroundWall { get; set; }

    /// <summary>
    /// Random generator used for determining the spawned blob types.
    /// </summary>
    private Random Random { get; }
    /// <summary>
    /// The physics world used for the simulation.
    /// </summary>
    private World World { get; }

    /// <summary>
    /// The type of the currently spawned blob.
    /// </summary>
    public int CurrentBlob { get; private set; }
    /// <summary>
    /// The type of the next blob to be spawned.
    /// </summary>
    public int NextBlob { get; private set; }
    /// <summary>
    /// The type of the currently held blob.
    /// </summary>
    public int HeldBlob => -1;
    /// <summary>
    /// Wether or not the player can currently spawn a blob. This is false when the last spawned blob is still falling.
    /// </summary>
    public bool CanSpawnBlob { get; private set; }

    /// <summary>
    /// The current score of the game
    /// </summary>
    public int Score { get; private set; }
    /// <summary>
    /// Flag indicating wether or not the game is over. The game is over when a blob "overfills" the arena.
    /// </summary>
    public bool IsGameOver { get; private set; }

    ///// <summary>
    ///// Stores the blobs that have collided this frame. This is used to prevent duplicate collisions.
    ///// </summary>
    //private HashSet<Blob> CollidedBlobs { get; }
    /// <summary>
    /// Stores all collision pairs that have occured this frame.
    /// </summary>
    private List<(Blob b0, Blob b1)> Collisions { get; }

    /// <summary>
    /// Keeps track of the last spawned blob. This is used to determine wether or not the player can spawn a new blob.
    /// </summary>
    private Blob? LastSpawned { get; set; }

    /// <summary>
    /// The rotation of the next blob to be spawned.
    /// </summary>
    public float SpawnRotation { get; private set; }

    /// <summary>
    /// Event that is fired when a blob is spawned.
    /// </summary>
    public event BlobEventHandler OnBlobSpawned;
    /// <summary>
    /// Event that is fired when a newly spawned blob collides for the first time.
    /// </summary>
    public event BlobEventHandler OnBlobPlaced;
    /// <summary>
    /// Event that is fired when two blobs combine.
    /// </summary>
    public event BlobEventHandler OnBlobsCombined;

    /// <summary>
    /// Event that is fired when a blob is destroyed. The argument is the type of the blob that was destroyed.
    /// </summary>
    public event BlobEventHandler OnBlobDestroyed;

    /// <summary>
    /// Event that is fired when the game is over.
    /// </summary>
    public event GameEventHandler OnGameOver;

    /// <summary>
    /// Creates a new simulation with the given seed.
    /// </summary>
    /// <param name="seed">The seed used to initialize the random blob generator.</param>
    public ClassicGameMode(int seed) {
        Blobs = new Dictionary<int, BlobData>();

        _GameObjects = new GameObjectCollection();

        Random = new Random(seed);
        World = new World(new Vector2(0, IGameMode.GRAVITY));

        //CollidedBlobs = new HashSet<Blob>();
        Collisions = new List<(Blob, Blob)>();

        LastSpawned = null;

        Score = 0;
        IsGameOver = false;
    }

    /// <summary>
    /// Initializes the simulation. Creates the arena walls physics bodies. Determines the first blob types to be spawned.
    /// </summary>
    public void Load() {
        TextResource blobDataText = ResourceManager.TextLoader.Get("blobs_classic");
        blobDataText.WaitForLoad();
        Blobs = blobDataText.Resource.Select(kvp => BlobData.Parse(kvp.Key, kvp.Value)).ToDictionary(d => d.Id, d => d);

        CurrentBlob = GenerateRandomBlobType();
        SpawnRotation = Random.NextSingle() * MathF.Tau;
        NextBlob = GenerateRandomBlobType();
        _GameObjects.AddRange(CreateArena(World));
        GroundWall = (GameObjects.FindByName("Ground") as Wall)!;

        CanSpawnBlob = true;
    }

    /// <summary>
    /// Used to update the game simulation. Is called every frame. Simulates the physics, handles game object adding and removing and check for game over conditions.
    /// </summary>
    /// <param name="dT"></param>
    public void Update(float dT) {
        if (IsGameOver)
            return;

        // Each tick is split into 50 smaller ticks to make the physics more stable.
        // 50 sub ticks is the sweet spot where the game feels imilar to the original.
        for (int i = 0; i < 50; i++) {
            World.Step(dT / 50f);
        }

        ResolveVeryCloseBlobs();

        ResolveBlobCollision();

        CheckGameOver();
    }

    /// <summary>
    /// Attempts to spawn a blob. Returns true if a blob was spawned, false otherwise.
    /// </summary>
    /// <param name="t">The t value where to spawn the blob above the arena. 0 is all the way left, 1 all the way right.</param>
    /// <param name="blob"></param>
    /// <returns>Returns true if the blob should be spawned; false otherwise.</returns>
    public bool TrySpawnBlob(float t) {
        if (!CanSpawnBlob || IsGameOver)
            return false;

        int type = CurrentBlob;

        float rot = SpawnRotation;
        SpawnRotation = Random.NextSingle() * MathF.Tau;
        CurrentBlob = NextBlob;
        NextBlob = GenerateRandomBlobType();

        float x = (t - 0.5f) * IGameMode.ARENA_WIDTH;
        float y = IGameMode.ARENA_SPAWN_Y_OFFSET;

        LastSpawned = CreateBlob(new Vector2(x, y), rot, type);
        CanSpawnBlob = false;

        OnBlobSpawned?.Invoke(this, new System.Numerics.Vector2(x, y), type);

        return true;
    }

    /// <summary>
    /// Attempts to hold the current blob. If a blob is already held, the current blob is swapped with the held blob.
    /// </summary>
    public void HoldBlob() {
        // Do nothing
    }

    private void ResolveVeryCloseBlobs() {
        const float EPSILON = 1f;

        foreach (Blob blob1 in GameObjects.OfType<Blob>().ToList()) {
            foreach (Blob blob2 in GameObjects.OfType<Blob>().ToList()) {

                if (blob1 == blob2)
                    continue;

                if (blob1.Type != blob2.Type)
                    continue;

                if (blob1.Data.ColliderData[0] != 'c' || blob2.Data.ColliderData[0] != 'c')
                    continue;

                if (Collisions.Contains((blob1, blob2)) || Collisions.Contains((blob2, blob1)))
                    continue;

                float distSqr = (blob1.Position - blob2.Position).Length();
                float b1Radius = float.Parse(blob1.Data.ColliderData[1..], CultureInfo.InvariantCulture);
                float b2Radius = float.Parse(blob2.Data.ColliderData[1..], CultureInfo.InvariantCulture);
                float r = b1Radius + b2Radius + EPSILON;

                if (distSqr > r)
                    continue;

                Collisions.Add((blob1, blob2));
            }
        }
    }

    /// <summary>
    /// Resolves any blob collisions that have occured this frame.
    /// </summary>
    private void ResolveBlobCollision() {
        foreach ((Blob b0, Blob b1) in Collisions) {
            if (!GameObjects.Contains(b0) || !GameObjects.Contains(b1))
                continue;

            Score += b0.Data.Score;

            Vector2 midPoint = (b0.Position + b1.Position) / 2f;
            RemoveBlob(b0);
            RemoveBlob(b1);

            if (b0.Data.MergeBlobId != -1)
                CreateBlob(midPoint, (b0.Rotation + b1.Rotation) / 2f, b0.Data.MergeBlobId);

            OnBlobsCombined?.Invoke(this, new System.Numerics.Vector2(b0.Position.X, b0.Position.Y), b0.Data.MergeBlobId);
        }
        Collisions.Clear();
        //CollidedBlobs.Clear();
    }

    /// <summary>
    /// Creates a new blob of the given type at the given position and adds it to the game.
    /// </summary>
    /// <param name="position"></param>
    /// <param name="type"></param>
    /// <returns></returns>
    private Blob CreateBlob(Vector2 position, float rotation, int type) {
        BlobData blobData = Blobs[type];
        Blob blob = new Blob(blobData, World, position, rotation);
        _GameObjects.Add(blob);
        blob.Body.OnCollision += OnBlobCollision;

        return blob;
    }

    /// <summary>
    /// Removes a blob from the game.
    /// </summary>
    /// <param name="blob"></param>
    private void RemoveBlob(Blob blob) {
        if (!GameObjects.Contains(blob))
            return;

        _GameObjects.Remove(blob);
        blob.Body.OnCollision -= OnBlobCollision;
        World.Remove(blob.Body);
    }

    /// <summary>
    /// Checks if the game's game over condition(s) are met and sets the game over flag accordingly.
    /// </summary>
    private void CheckGameOver() {
        GameObjects.Enumerate(gO => {
            bool tmp = gO is Blob blob && blob.Position.Y <= 0 && blob != LastSpawned;
            IsGameOver |= tmp;
        });

        if (IsGameOver)
            OnGameOver?.Invoke(this);
    }

    /// <summary>
    /// Callback from the physics engine to handle collisions between blobs and/or walls.
    /// </summary>
    /// <param name="sender">The object initializing the collision.</param>
    /// <param name="other">The other object involved in the collision.</param>
    /// <param name="contact">The contact data of the collision.</param>
    /// <returns>???? No idea. Everyone uses true everywhere. Can't find any docs for it either.</returns>
    private bool OnBlobCollision(Fixture sender, Fixture other, Contact contact) {
        if (IsGameOver)
            return true;

        CheckForBlobSpawnReenable(sender, other);

        Blob? senderBlob = sender.Body.Tag as Blob;
        Blob? otherBlob = other.Body.Tag as Blob;

        if (senderBlob == null || otherBlob == null)
            return true;

        if (senderBlob.Type != otherBlob.Type)
            return true;

        //if (CollidedBlobs.Contains(senderBlob) || CollidedBlobs.Contains(otherBlob))
        if (Collisions.Contains((senderBlob, otherBlob)) || Collisions.Contains((otherBlob, senderBlob)))
            return true;

        //CollidedBlobs.Add(senderBlob);
        //CollidedBlobs.Add(otherBlob);
        Collisions.Add((senderBlob, otherBlob));

        return true;
    }

    /// <summary>
    /// Checks if the last spawned blob has collided with either the ground wall or another blob. If it is, reenables blob spawning.
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="other"></param>
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
            int lastSpawnedType = LastSpawned!.Type;
            CanSpawnBlob = true;
            LastSpawned = null;

            Blob blob = sender.Body.Tag is Blob b0 ? b0 : (Blob)other.Body.Tag;

            OnBlobPlaced?.Invoke(this, new System.Numerics.Vector2(blob.Position.X, blob.Position.Y), lastSpawnedType);
        }
    }

    /// <summary>
    /// Generates a random blob type up to the maximum spawnable blob type.
    /// </summary>
    /// <returns></returns>
    private int GenerateRandomBlobType() {
        float totalSpawnWeight = Blobs.Values.Sum(b => b.SpawnWeight);
        float spawnValue = Random.NextSingle() * totalSpawnWeight;

        foreach (BlobData blobData in Blobs.Values) {
            spawnValue -= blobData.SpawnWeight;
            if (spawnValue <= 0)
                return blobData.Id;
        }

        return 0;
    }

    /// <summary>
    /// Creates the arena walls and returns them.
    /// </summary>
    /// <param name="world">The physics engine world</param>
    /// <returns>Returns an enumerable with the wall game objects.</returns>
    private static IEnumerable<GameObject> CreateArena(World world) {
        const float WALL_HEIGHT_EXTENSION = 400;
        float totalWidth = IGameMode.ARENA_WIDTH + 2 * IGameMode.ARENA_WALL_THICKNESS;

        float x = -totalWidth / 2;
        float y = 0;

        GameObject[] walls = new GameObject[] {
            new Wall("Ground", world, new Rectangle(x, y + IGameMode.ARENA_HEIGHT, totalWidth, IGameMode.ARENA_WALL_THICKNESS)),
            new Wall("Left Wall", world, new Rectangle(x, y - WALL_HEIGHT_EXTENSION, IGameMode.ARENA_WALL_THICKNESS, IGameMode.ARENA_HEIGHT + WALL_HEIGHT_EXTENSION)),
            new Wall("Right Wall", world, new Rectangle(x + IGameMode.ARENA_WIDTH + IGameMode.ARENA_WALL_THICKNESS, y - WALL_HEIGHT_EXTENSION, IGameMode.ARENA_WALL_THICKNESS, IGameMode.ARENA_HEIGHT + WALL_HEIGHT_EXTENSION))
        };

        return walls;
    }
}
