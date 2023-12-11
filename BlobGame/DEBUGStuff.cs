using Raylib_CsLo;
using System.Diagnostics;
using System.Numerics;
using System.Text.Json;

namespace BlobGame;
internal static class DEBUGStuff {

    public static void DEBUG_SerializeText() {
        Dictionary<string, string> scrolls = new Dictionary<string, string> {
            { "Classic", "Play the classic mode!\nCombine pieces and gain points." },
        };
        string json = JsonSerializer.Serialize(scrolls, new JsonSerializerOptions() { WriteIndented = true });
        File.WriteAllText(@"G:\Coding\C#\BlobGame\BlobGame\Resources\References\Texts\game_mode_descriptions.json", json);

        Dictionary<string, string> d2 = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(@"G:\Coding\C#\BlobGame\BlobGame\Resources\References\Texts\game_mode_descriptions.json"));
    }

    public static void DEBUG_PrintNPatchJson() {
        NPatchInfo npi = new NPatchInfo(new Rectangle(0, 0, 500, 400), 65, 50, 856, 167, NPatchLayout.NPATCH_NINE_PATCH);

        string json = JsonSerializer.Serialize<Dictionary<string, int>>(new() {
            { "left", npi.left },
            { "right", npi.right },
            { "top", npi.top },
            { "bottom", npi.bottom },
            { "layout", npi.layout },
        }, new JsonSerializerOptions() { WriteIndented = true });

        Dictionary<string, int>? dict = JsonSerializer.Deserialize<Dictionary<string, int>>(json);
        NPatchInfo npi2 = new NPatchInfo(new Rectangle(0, 0, 500, 400), dict["left"], dict["right"], dict["top"], dict["bottom"], (NPatchLayout)dict["layout"]);
    }

    public static void DEBUG_ConvertColliderCoords() {
        string str = @"145,1,
1294,1,

1392,44,

1440,149,
1440,1291,

1392,1387,

1284,1440,
155,1440,

42,1390,
0,1291,
0,150,

53,40";

        // The size of the vertices in texture coords
        const float OLD_SIZE = 1440f;
        // The size of the vertices in game coords
        const float NEW_SIZE = 360f;

        Vector2[] vertices = str.Replace("\r", "").Split('\n', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).Select(l => {
            string[] c = l.Replace(',', ';').Split(';');
            return new Vector2(float.Parse(c[0]), float.Parse(c[1]));
        }).ToArray();

        Vector2[] newVertices = vertices.Select(v => v / OLD_SIZE * NEW_SIZE - new Vector2(NEW_SIZE / 2f, NEW_SIZE / 2f)).ToArray();

        Debug.WriteLine(string.Join(",", newVertices.Select(v => $"{v.X},{v.Y}")));
    }
}
