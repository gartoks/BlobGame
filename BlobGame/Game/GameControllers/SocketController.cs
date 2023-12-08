using BlobGame.Game.GameModes;
using BlobGame.Game.GameObjects;
using System.Diagnostics;
using System.Net.Sockets;
using System.Text;

namespace BlobGame.Game.GameControllers;

internal class SocketController : IGameController {

    /// <summary>
    /// The index of the running game.
    /// </summary>
    private int GameIndex { get; }

    /// <summary>
    /// The port used to connect.
    /// </summary>
    private int Port { get; }

    /// <summary>
    /// The tcp client used to control this controller.
    /// </summary>
    private TcpClient? Client { get; set; }
    /// <summary>
    /// The network stream used to communicate with the server.
    /// </summary>
    private NetworkStream? Stream { get; set; }

    internal bool IsConnected => Client != null && Client.Connected;

    private (float t, bool shouldDrop, bool shouldHold)? FrameInputs { get; set; }

    public SocketController(int gameIndex, int port) {
        GameIndex = gameIndex;
        Port = port;
    }

    ~SocketController() {
        Close();
    }

    public void Load() {
        try {
            Client = new TcpClient("localhost", Port);
            Stream = Client.GetStream();
        } catch (SocketException) {
            Client = null;
            Stream = null!;
            Debug.WriteLine("Controller stream was closed.");
        }
        if (Client != null)
            Debug.WriteLine($"Connected to localhost:{Port}");
    }

    public void Close() {
        Debug.WriteLine("Closing tcp socket");
        Client?.Close();
    }

    /// <summary>
    /// Retrieves the current value of t, which represents the position of the dropper above the arena.
    /// </summary>
    /// <returns>The current value of t.</returns>
    public float GetCurrentT() {
        return FrameInputs == null ? -1 : FrameInputs.Value.t;
    }

    /// <summary>
    /// Attempts to spawn a blob in the provided game simulation.
    /// </summary>
    /// <param name="simulation">The game simulation in which to spawn the blob.</param>
    /// <param name="t">The t value at which the blob is spawned, which represents the position of the dropper above the arena..</param>
    /// <returns>True if blob spawning was attempted, otherwise false.</returns>
    public bool SpawnBlob(IGameMode simulation, out float t) {
        t = GetCurrentT();

        if (!simulation.CanSpawnBlob)
            return false;

        return FrameInputs != null && FrameInputs.Value.shouldDrop;
    }

    public bool HoldBlob() {
        return FrameInputs != null && FrameInputs.Value.shouldHold;
    }

    public void Update(float dT, IGameMode simulation) {
        if (!IsConnected)
            return;

        // send simulation state
        SendGameState(simulation);

        // wait for inputs
        ReceiveInputs();
    }

    public void SendGameState(IGameMode simulation) {
        List<Blob> blobs = new();
        simulation.GameObjects.Enumerate(go => {
            if (go.GetType() == typeof(Blob)) {
                blobs.Add((Blob)go);
            }
        });

        string gameModeKey = IGameMode.GameModeTypes.Where(k => k.Value == simulation.GetType()).Select(k => k.Key).Single();


        IEnumerable<byte> buffer = new byte[0];
        buffer = buffer.Concat(BitConverter.GetBytes(blobs.Count));
        foreach (Blob blob in blobs) {
            buffer = buffer.Concat(BitConverter.GetBytes(blob.Position.X));
            buffer = buffer.Concat(BitConverter.GetBytes(blob.Position.Y));
            buffer = buffer.Concat(BitConverter.GetBytes(blob.Type));
        }
        buffer = buffer.Concat(BitConverter.GetBytes(simulation.CurrentBlob));
        buffer = buffer.Concat(BitConverter.GetBytes(simulation.NextBlob));
        buffer = buffer.Concat(BitConverter.GetBytes(simulation.HeldBlob));
        buffer = buffer.Concat(BitConverter.GetBytes(simulation.Score));
        buffer = buffer.Concat(BitConverter.GetBytes(GameIndex));
        buffer = buffer.Concat(new byte[] { simulation.CanSpawnBlob ? (byte)0b00000001 : (byte)0b00000000 });
        buffer = buffer.Concat(new byte[] { simulation.IsGameOver ? (byte)0b00000001 : (byte)0b00000000 });
        byte[] gameModeKeyData = Encoding.UTF8.GetBytes(gameModeKey == null ? "" : gameModeKey);
        buffer = buffer.Concat(BitConverter.GetBytes(gameModeKeyData.Length));
        buffer = buffer.Concat(gameModeKeyData);

        bool failed = false;
        try {
            Stream?.Write(buffer.ToArray());
        } catch (SocketException) {
            failed = true;
        } catch (IOException) {
            failed = true;
        } catch (ObjectDisposedException) {
            failed = true;
        }

        if (failed) {
            Debug.WriteLine("Controller stream was closed.");
            return;
        }
    }

    /// <summary>
    /// Waits for the next frame of inputs and stores them in frameInputs.
    /// </summary>
    public void ReceiveInputs() {
        byte[] buffer = new byte[6];
        bool failed = false;
        try {
            Stream?.ReadExactly(buffer, 0, 6);
        } catch (EndOfStreamException) {
            failed = true;
        } catch (SocketException) {
            failed = true;
        } catch (IOException) {
            failed = true;
        } catch (ObjectDisposedException) {
            failed = true;
        }
        if (failed) {
            Debug.WriteLine("Controller input stream was closed.");
            return;
        }

        float t = BitConverter.ToSingle(buffer, 0);
        bool shouldDrop = BitConverter.ToBoolean(buffer, sizeof(float));
        bool shouldHold = (buffer[sizeof(float) + sizeof(bool)] & 0b1) == 0b1;
        FrameInputs = (t, shouldDrop, shouldHold);
    }
}
