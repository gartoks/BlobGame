using BlobGame.Game.GameModes;
using BlobGame.Game.GameObjects;
using System.Diagnostics;
using System.IO.Compression;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;

namespace BlobGame.Game.GameControllers;

internal class SocketController : IGameController {

    /// <summary>
    /// The index of the running game.
    /// </summary>
    private int GameIndex { get; }

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

    /// <summary>
    /// The global listener waiting for clients to connect.
    /// </summary>
    private static TcpListener? Listener = null;

    public static void Load(int port){
        if (Listener == null){
            Listener = new TcpListener(IPAddress.Any, port);
        
            Listener.Start();
        }
    }

    public static bool HasPendingConnections => Listener != null && Listener.Pending();
    
    public static void Unload(){
        if (Listener != null){
            Listener.Stop();
            Listener = null;
        }
    }

    public SocketController(int gameIndex) {
        GameIndex = gameIndex;
    }

    public void Load(){
        Client = Listener?.AcceptTcpClient();

        if (Client != null && Client.Connected)
            Console.WriteLine($"Got connection for game {GameIndex}");
     
        Stream = Client?.GetStream();
    }

    ~SocketController() {
        Close();
    }

    public void Close() {
        Debug.WriteLine("Closing tcp socket");
        Stream?.Close();
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

        FramePacket packet = new FramePacket
        {
            BlobCount = blobs.Count(),
            CurrentBlobType = simulation.CurrentBlob,
            NextBlobType = simulation.NextBlob,
            HeldBlobType = simulation.HeldBlob,
            CurrentScore = simulation.Score,
            GameIndex = GameIndex,
            CanSpawnBlob = simulation.CanSpawnBlob,
            IsGameOver = simulation.IsGameOver,
            GameModeKey = gameModeKey
        };
        NetworkBlob[] networkBlobs = blobs.Select(blob => new NetworkBlob{
            Type = blob.Type,
            x = blob.Position.X,
            y = blob.Position.Y,
        }).ToArray();


        byte[] packetBytes = getBytes(packet);
        byte[] blobBytes = getBytes(networkBlobs);


        bool failed = false;
        try {
            Stream?.Write(
                BitConverter.GetBytes((int)(packetBytes.Length + blobBytes.Length))
                    .Concat(packetBytes)
                    .Concat(blobBytes)
                    .ToArray());
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

        InputPacket packet = fromBytes<InputPacket>(buffer);
        FrameInputs = (packet.t, packet.ShouldDrop, packet.ShouldHold);
    }

    private byte[] Compress(byte[] data){
            using (var compressedStream = new MemoryStream())
            using (var zipStream = new GZipStream(compressedStream, CompressionMode.Compress)){
                zipStream.Write(data, 0, data.Length);
                return compressedStream.ToArray();
            }
        }

    private byte[] Decompress(byte[] data){
        using (var compressedStream = new MemoryStream(data))
        using (var zipStream = new GZipStream(compressedStream, CompressionMode.Decompress))
        using (var resultStream = new MemoryStream()){
            zipStream.CopyTo(resultStream);
            return resultStream.ToArray();
        }
    }

    private byte[] getBytes<T>(T str) where T : struct {
        int size = Marshal.SizeOf(str);
        byte[] arr = new byte[size];

        IntPtr ptr = IntPtr.Zero;
        try{
            ptr = Marshal.AllocHGlobal(size);
            Marshal.StructureToPtr(str, ptr, true);
            Marshal.Copy(ptr, arr, 0, size);
        }
        finally{
            Marshal.FreeHGlobal(ptr);
        }
        return arr;
    }
    private byte[] getBytes<T>(T[] array) where T: struct{
        return MemoryMarshal.AsBytes(array.AsSpan()).ToArray();
    }

    T fromBytes<T>(byte[] arr) where T: struct {
        T str = new T();

        int size = Marshal.SizeOf(str);
        IntPtr ptr = IntPtr.Zero;
        try{
            ptr = Marshal.AllocHGlobal(size);

            Marshal.Copy(arr, 0, ptr, size);

            str = (T)Marshal.PtrToStructure(ptr, typeof(T));
        }
        finally{
            Marshal.FreeHGlobal(ptr);
        }
        return str;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode, Pack = 1)]
    private struct FramePacket{
        public int BlobCount;
        public int CurrentBlobType;
        public int NextBlobType;
        public int HeldBlobType;
        public int CurrentScore;
        public int GameIndex;
        [MarshalAs(UnmanagedType.I1)]
        public bool CanSpawnBlob;
        [MarshalAs(UnmanagedType.I1)]
        public bool IsGameOver;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 10)]
        public string GameModeKey;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private struct NetworkBlob{
        public float x, y;
        public int Type;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private struct InputPacket{
        public float t;
        [MarshalAs(UnmanagedType.I1)]
        public bool ShouldDrop, ShouldHold;
    }

    
}
