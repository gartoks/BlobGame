namespace BlobGame.Game;
internal class BlobData {
    public static IReadOnlyList<(string name, eBlobType type, int score, float radius, string textureKey)> Data { get; }
        = new (string name, eBlobType type, int score, float radius, string textureKey)[] {
            ("Cherry", eBlobType.Drone, 1, 22.5f, "0"),
            ("Strawberry", eBlobType.Heart, 3, 30f, "1"),
            ("Grape", eBlobType.Grape, 6, 41f, "2"),
            ("Orange", eBlobType.Orange, 10, 47f, "3"),
            ("Tomato", eBlobType.Tomato, 15, 61f, "4"),
            ("Apple", eBlobType.Apple, 21, 79f, "5"),
            ("Yuzu", eBlobType.Yuzu, 28, 90f, "6"),
            ("Peach", eBlobType.Peach, 36, 109.5f, "7"),
            ("Pineapple", eBlobType.Pineapple, 45, 123f, "8"),
            ("Honeydew", eBlobType.Honeydew, 55, 152.5f, "9"),
            ("Watermelon", eBlobType.Watermelon, 0, 180f, "10"),
    };

}
