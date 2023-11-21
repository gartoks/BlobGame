﻿using BlobGame.Game.GameModes;
using BlobGame.Game.GameObjects;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;

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

    private (float t, bool shouldDrop)? FrameInputs { get; set; }

    /// <summary>
    /// The global listener waiting for clients to connect.
    /// </summary>
    private static TcpListener? Listener = null;

    public static void Load(int port){
        if (Listener == null){
            Listener = new TcpListener(IPAddress.Loopback, port);
        
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
        t = -1;
        if (!simulation.CanSpawnBlob)
            return false;

        t = GetCurrentT();

        return FrameInputs != null && FrameInputs.Value.shouldDrop;
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

        IEnumerable<byte> buffer = new byte[0];
        buffer = buffer.Concat(BitConverter.GetBytes(blobs.Count));
        foreach (Blob blob in blobs) {
            buffer = buffer.Concat(BitConverter.GetBytes(blob.Position.X));
            buffer = buffer.Concat(BitConverter.GetBytes(blob.Position.Y));
            buffer = buffer.Concat(BitConverter.GetBytes((int)blob.Type));
        }
        buffer = buffer.Concat(BitConverter.GetBytes((int)simulation.CurrentBlob));
        buffer = buffer.Concat(BitConverter.GetBytes((int)simulation.NextBlob));
        buffer = buffer.Concat(BitConverter.GetBytes(simulation.Score));
        buffer = buffer.Concat(BitConverter.GetBytes(GameIndex));
        buffer = buffer.Concat(BitConverter.GetBytes(simulation.CanSpawnBlob));
        buffer = buffer.Concat(BitConverter.GetBytes(simulation.IsGameOver));
        //buffer = buffer.Concat(BitConverter.GetBytes((int)simulation.GameMode));

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
        byte[] buffer = new byte[5];
        bool failed = false;
        try {
            Stream?.ReadExactly(buffer, 0, 5);
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
        bool shouldDrop = BitConverter.ToBoolean(buffer, 4);
        FrameInputs = (t, shouldDrop);
    }
}
