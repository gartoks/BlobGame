using nkast.Aether.Physics2D.Common;
using nkast.Aether.Physics2D.Dynamics;

namespace BlobGame.Game.Blobs;


public record BlobData {
    public static BlobData Parse(string idStr, string dataStr) {
        int id = int.Parse(idStr);
        string[] data = dataStr.Split(';', StringSplitOptions.TrimEntries);

        string name = data[0];
        int score = int.Parse(data[1]);
        int mergeBlobId = int.Parse(data[2]);
        int mergeWithBlobId = int.Parse(data[3]);
        float shatterSpeed = float.Parse(data[4]);
        float mass = float.Parse(data[5]);
        string textureKey = data[6];

        string[] originComponents = data[7].Split(',', StringSplitOptions.TrimEntries);
        if (originComponents.Length != 2)
            throw new Exception("Invalid texture offset data. There must be exactly two components.");
        Vector2 origin = new Vector2(
            float.Parse(originComponents[0]),
            float.Parse(originComponents[1]));

        string[] textureScaleComponents = data[8].Split(',', StringSplitOptions.TrimEntries);
        if (textureScaleComponents.Length != 2)
            throw new Exception("Invalid texture scale data. There must be exactly two components.");
        Vector2 textureScale = new Vector2(
            float.Parse(textureScaleComponents[0]),
            float.Parse(textureScaleComponents[1]));

        string colliderData = data[9];

        return new BlobData(id, name, score, mergeBlobId, mergeWithBlobId, colliderData, shatterSpeed, mass, textureKey, origin, textureScale);
    }

    public int Id { get; }
    public string Name { get; }
    public int Score { get; }
    public int MergeBlobId { get; }
    public int MergeWithBlobId { get; }
    private string ColliderData { get; }
    public float ShatterSpeed { get; }
    public float Mass { get; }
    public string TextureKey { get; }
    public Vector2 Origin { get; }
    public Vector2 TextureScale { get; }

    private BlobData(int id, string name, int score, int mergeBlobId, int mergeWithBlobId,
                     string colliderData, float shatterSpeed, float mass,
                     string textureKey, Vector2 origin, Vector2 textureScale) {

        Id = id;
        Name = name;
        Score = score;
        MergeBlobId = mergeBlobId;
        MergeWithBlobId = mergeWithBlobId;
        ColliderData = colliderData;
        ShatterSpeed = shatterSpeed;
        Mass = mass;
        TextureKey = textureKey;
        Origin = origin;
        TextureScale = textureScale;
    }

    internal bool AsCircle(out float radius) {
        if (ColliderData[0] != 'c') {
            radius = 0;
            return false;
        }

        string[] colliderValues = ColliderData[1..].Split(',', StringSplitOptions.TrimEntries);
        radius = float.Parse(colliderValues[0]) / GameObject.POSITION_MULTIPLIER;
        return true;
    }

    internal bool AsPolygon(out Vertices vertices) {
        if (ColliderData[0] != 'p') {
            vertices = null!;
            return false;
        }

        string[] colliderValues = ColliderData[1..].Split(',', StringSplitOptions.TrimEntries);
        vertices = new Vertices(ParseVertices(Origin, colliderValues));
        return true;
    }

    internal Fixture CreateFixture(Body body) {
        char colliderType = ColliderData[0];
        string[] colliderValues = ColliderData[1..].Split(',', StringSplitOptions.TrimEntries);

        switch (colliderType) {
            case 'c':
                float radius = float.Parse(colliderValues[0]) / GameObject.POSITION_MULTIPLIER;
                return body.CreateCircle(radius, 1);
            case 'p':
                Vertices vertices = new Vertices(ParseVertices(Origin, colliderValues));
                return body.CreatePolygon(vertices, 1);
            default:
                throw new Exception($"Unknown collider type '{colliderType}'");
        }
    }

    private static IEnumerable<Vector2> ParseVertices(Vector2 origin, string[] colliderValues) {
        if (colliderValues.Length % 2 != 0)
            throw new Exception("Invalid collider data. There must be an even number of collider values for a polygon type collider.");

        List<Vector2> vertices = new List<Vector2>(colliderValues.Length / 2);
        for (int i = 0; i < colliderValues.Length; i += 2) {
            float x = float.Parse(colliderValues[i]) / GameObject.POSITION_MULTIPLIER;
            float y = float.Parse(colliderValues[i + 1]) / GameObject.POSITION_MULTIPLIER;
            vertices.Add(new Vector2(x, y));
        }

        return vertices;
    }
}
