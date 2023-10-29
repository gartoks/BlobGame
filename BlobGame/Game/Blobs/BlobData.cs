namespace BlobGame.Game.Blobs;
/// <summary>
/// Class to hold blob specific data.
/// </summary>
internal class BlobData {
    public static IReadOnlyList<(string name, eBlobType type, int score, float radius, float mass, string textureKey)> Data { get; }
        = new (string name, eBlobType type, int score, float radius, float mass, string textureKey)[] {
            ("Cherry", eBlobType.Cherry, 1, 22.5f, 1f, "0"),
            ("Strawberry", eBlobType.Strawberry, 3, 30f, 2f, "1"),
            ("Grape", eBlobType.Heart, 6, 41f, 3f, "2"),
            ("Orange", eBlobType.Cookie, 10, 47f, 4f, "3"),
            ("Tomato", eBlobType.Doughnut, 15, 61f, 5f, "4"),
            ("Apple", eBlobType.Apple, 21, 79f, 6f, "5"),
            ("Yuzu", eBlobType.Yuzu, 28, 90f, 7f, "6"),
            ("Peach", eBlobType.Peach, 36, 109.5f, 8f, "7"),
            ("Pineapple", eBlobType.Pineapple, 45, 123f, 9f, "8"),
            ("Honeydew", eBlobType.Honeydew, 55, 152.5f, 10f, "9"),
            ("Watermelon", eBlobType.Watermelon, 0, 180f,11f, "10"),
    };

}
