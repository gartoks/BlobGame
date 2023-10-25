namespace BlobGame.Game.Blobs;
/// <summary>
/// Class to hold blob specific data.
/// </summary>
internal class BlobData {
    public static IReadOnlyList<(string name, eBlobType type, int score, float radius, string textureKey)> Data { get; }
        = new (string name, eBlobType type, int score, float radius, string textureKey)[] {
            ("Cherry", eBlobType.Cherry, 1, 22.5f, "0"),
            ("Strawberry", eBlobType.Strawberry, 3, 30f, "1"),
            ("Grape", eBlobType.Heart, 6, 41f, "2"),
            ("Orange", eBlobType.Cookie, 10, 47f, "3"),
            ("Tomato", eBlobType.Doughnut, 15, 61f, "4"),
            ("Apple", eBlobType.Apple, 21, 79f, "5"),
            ("Yuzu", eBlobType.Yuzu, 28, 90f, "6"),
            ("Peach", eBlobType.Peach, 36, 109.5f, "7"),
            ("Pineapple", eBlobType.Pineapple, 45, 123f, "8"),
            ("Honeydew", eBlobType.Honeydew, 55, 152.5f, "9"),
            ("Watermelon", eBlobType.Watermelon, 0, 180f, "10"),
    };

}
