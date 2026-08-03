import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:image/image.dart' as img;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as status;

void main() {
  runApp(const TraductorSenasApp());
}

class TraductorSenasApp extends StatelessWidget {
  const TraductorSenasApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Traductor de Señas',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF3F4F8),
        fontFamily: 'Roboto',
      ),
      // La app arranca pidiendo el permiso de cámara. Recién al tocar
      // "Permitir acceso" (o "Ahora no") se pasa a la pantalla principal.
      home: const CameraPermissionScreen(),
    );
  }
}

// =========================================================================
// PANTALLA: SOLICITUD DE PERMISO DE CÁMARA
// =========================================================================

class CameraPermissionScreen extends StatelessWidget {
  const CameraPermissionScreen({super.key});

  void _continuar(BuildContext context, {required bool permisoConcedido}) {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(
        builder: (_) => TraductorScreen(permisoConcedido: permisoConcedido),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 420),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 28, vertical: 40),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(28),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.05),
                      blurRadius: 20,
                      offset: const Offset(0, 8),
                    ),
                  ],
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Círculo lila con ícono de cámara
                    Container(
                      width: 88,
                      height: 88,
                      decoration: const BoxDecoration(
                        color: Color(0xFFE7E6F7),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.photo_camera_outlined,
                        size: 38,
                        color: Color(0xFF3E5C8A),
                      ),
                    ),
                    const SizedBox(height: 24),

                    const Text(
                      'Necesitamos acceso a tu cámara',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 19,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFF1C1C28),
                      ),
                    ),
                    const SizedBox(height: 12),

                    const Text(
                      'La usamos para reconocer tus señas en tiempo '
                      'real. No se graba ni almacena sin tu permiso.',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 13.5,
                        height: 1.4,
                        color: Color(0xFF8A8A94),
                      ),
                    ),
                    const SizedBox(height: 28),

                    SizedBox(
                      width: double.infinity,
                      height: 50,
                      child: ElevatedButton(
                        onPressed: () =>
                            _continuar(context, permisoConcedido: true),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF3E5C8A),
                          foregroundColor: Colors.white,
                          elevation: 0,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14),
                          ),
                        ),
                        child: const Text(
                          'Permitir acceso',
                          style: TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 14),

                    TextButton(
                      onPressed: () =>
                          _continuar(context, permisoConcedido: false),
                      child: const Text(
                        'Ahora no',
                        style: TextStyle(
                          fontSize: 14,
                          color: Color(0xFF9B9BA5),
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// =========================================================================
// PANTALLA PRINCIPAL: TRADUCTOR
// =========================================================================

class TraductorScreen extends StatefulWidget {
  final bool permisoConcedido;

  const TraductorScreen({super.key, required this.permisoConcedido});

  @override
  State<TraductorScreen> createState() => _TraductorScreenState();
}

class _TraductorScreenState extends State<TraductorScreen> {
  static const _wsUrl = String.fromEnvironment('WS_URL');
  bool _traduciendo = false;
  String _textoTraduccion = 'Presiona iniciar para traducir';
  double _confianza = 0.0;

  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  bool _isConnected = false;
  CameraController? _cameraController;
  bool _cameraLista = false;
  bool _enviandoFrame = false;
  int _ultimoFrameMs = 0;

  // Texto a voz: dice en voz alta cada palabra reconocida.
  final FlutterTts _tts = FlutterTts();
  bool _sonidoActivado = true;

  @override
  void initState() {
    super.initState();
    _configurarTts();
    if (_permisoCamara) _inicializarCamara();
  }

  Future<void> _inicializarCamara() async {
    if (_cameraLista) return;
    try {
      final cameras = await availableCameras();
      final frontal = cameras.where((c) => c.lensDirection == CameraLensDirection.front);
      final camera = frontal.isNotEmpty ? frontal.first : cameras.first;
      final controller = CameraController(
        camera,
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() {
        _cameraController = controller;
        _cameraLista = true;
      });
    } on CameraException catch (e) {
      if (mounted) {
        setState(() {
          _permisoCamara = false;
          _textoTraduccion = 'No se pudo usar la cámara: ${e.description ?? e.code}';
        });
      }
    }
  }

  Future<void> _configurarTts() async {
    await _tts.setLanguage('es-ES');
    await _tts.setSpeechRate(0.45); // un poco más lento, se entiende mejor
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);
  }

  void _toggleSonido() {
    setState(() => _sonidoActivado = !_sonidoActivado);
    if (!_sonidoActivado) {
      _tts.stop();
    }
  }

  // Estado local del permiso, arranca con lo que trajo la pantalla
  // anterior pero se puede actualizar acá mismo si el usuario lo
  // concede más tarde desde el aviso.
  late bool _permisoCamara = widget.permisoConcedido;

  void _toggleTraduccion() {
    if (_traduciendo) {
      _desconectar();
      return;
    }

    if (!_permisoCamara) {
      _mostrarAvisoPermiso();
      return;
    }

    _conectar();
  }

  void _mostrarAvisoPermiso() {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(18),
        ),
        title: const Text(
          'Falta el permiso de cámara',
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 17),
        ),
        content: const Text(
          'Para traducir tus señas en tiempo real necesitamos acceso '
          'a la cámara. Actívalo para poder empezar.',
          style: TextStyle(color: Color(0xFF6B6B76)),
        ),
        actionsPadding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancelar'),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF3E5C8A),
              foregroundColor: Colors.white,
              elevation: 0,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            onPressed: () {
              Navigator.of(context).pop(); // cierra el diálogo
              setState(() => _permisoCamara = true);
              _inicializarCamara().then((_) => _conectar());
            },
            child: const Text('Dar permiso'),
          ),
        ],
      ),
    );
  }

  void _conectar() {
    if (!_cameraLista || _cameraController == null) {
      _inicializarCamara();
      return;
    }
    if (_wsUrl.isEmpty) {
      setState(() => _textoTraduccion = 'Falta configurar la URL del servidor');
      return;
    }
    setState(() {
      _traduciendo = true;
      _textoTraduccion = 'Conectando al servidor...';
      _confianza = 0.0;
    });

    try {
      _channel = WebSocketChannel.connect(
        Uri.parse(_wsUrl),
      );

      _subscription = _channel!.stream.listen(
        (data) {
          if (data is String) {
            try {
              final json = jsonDecode(data) as Map<String, dynamic>;
              final type = json['type'] as String?;

              if (type == 'prediction') {
                final palabra = json['palabra'] ?? '';
                final confianza = (json['confianza'] ?? 0.0) as double;
                if (palabra.isNotEmpty && mounted) {
                  setState(() {
                    _textoTraduccion = palabra;
                    _confianza = confianza;
                  });
                  if (_sonidoActivado) {
                    // No usamos await: no queremos bloquear la llegada
                    // de los próximos mensajes del socket mientras habla.
                    _tts.speak(palabra);
                  }
                }
              }
            } catch (e) {
              print('Error procesando mensaje: $e');
            }
          }
        },
        onDone: () {
          if (mounted && _traduciendo) {
            setState(() {
              _traduciendo = false;
              _textoTraduccion = 'Servidor desconectado';
              _confianza = 0.0;
              _isConnected = false;
            });
          }
        },
        onError: (error) {
          if (mounted) {
            setState(() {
              _traduciendo = false;
              _textoTraduccion = 'Error: $error';
              _confianza = 0.0;
              _isConnected = false;
            });
          }
        },
      );

      setState(() {
        _textoTraduccion = 'Esperando señas...';
        _isConnected = true;
      });
      _iniciarEnvioFrames();
    } catch (e) {
      setState(() {
        _traduciendo = false;
        _textoTraduccion = 'No se pudo conectar: $e';
        _confianza = 0.0;
        _isConnected = false;
      });
    }
  }

  void _iniciarEnvioFrames() {
    final controller = _cameraController;
    if (controller == null || controller.value.isStreamingImages) return;
    controller.startImageStream(_enviarFrameAlServidor);
  }

  void _detenerEnvioFrames() {
    final controller = _cameraController;
    if (controller != null && controller.value.isStreamingImages) {
      controller.stopImageStream();
    }
  }

  void _enviarFrameAlServidor(CameraImage frame) {
    final ahora = DateTime.now().millisecondsSinceEpoch;
    if (!_traduciendo || !_isConnected || _enviandoFrame || ahora - _ultimoFrameMs < 250) {
      return;
    }
    _enviandoFrame = true;
    _ultimoFrameMs = ahora;
    try {
      final jpeg = _cameraImageAJpeg(frame);
      _channel?.sink.add(jpeg);
    } catch (e) {
      debugPrint('No se pudo enviar el frame: $e');
    } finally {
      _enviandoFrame = false;
    }
  }

  List<int> _cameraImageAJpeg(CameraImage frame) {
    final image = img.Image(width: frame.width, height: frame.height);
    final yPlane = frame.planes[0];
    final uPlane = frame.planes[1];
    final vPlane = frame.planes[2];
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;

    for (var y = 0; y < frame.height; y++) {
      final yRow = y * yPlane.bytesPerRow;
      final uvRow = (y >> 1) * uPlane.bytesPerRow;
      for (var x = 0; x < frame.width; x++) {
        final yValue = yPlane.bytes[yRow + x];
        final uvIndex = uvRow + (x >> 1) * uvPixelStride;
        final uValue = uPlane.bytes[uvIndex] - 128;
        final vValue = vPlane.bytes[uvIndex] - 128;
        final r = (yValue + 1.402 * vValue).round().clamp(0, 255);
        final g = (yValue - 0.344136 * uValue - 0.714136 * vValue).round().clamp(0, 255);
        final b = (yValue + 1.772 * uValue).round().clamp(0, 255);
        image.setPixelRgb(x, y, r, g, b);
      }
    }
    return img.encodeJpg(img.copyResize(image, width: 320), quality: 70);
  }

  void _desconectar() {
    _detenerEnvioFrames();
    try {
      _channel?.sink.add(jsonEncode({'type': 'stop'}));
    } catch (_) {}

    _subscription?.cancel();
    _subscription = null;
    _channel?.sink.close(status.goingAway);
    _channel = null;

    setState(() {
      _traduciendo = false;
      _textoTraduccion = 'Presiona iniciar para traducir';
      _confianza = 0.0;
      _isConnected = false;
    });
  }

  @override
  void dispose() {
    _desconectar();
    _cameraController?.dispose();
    _tts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        body: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 420),
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const SizedBox(height: 8),
                  // Título en negrita
                  const Text(
                    'Traductor de Señas',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF1C1C28),
                    ),
                  ),
                  const SizedBox(height: 20),
                  _CameraPreview(
                    activo: _traduciendo,
                    isConnected: _isConnected,
                    controller: _cameraController,
                  ),
                  const SizedBox(height: 20),
                  _BotonTraduccion(
                    activo: _traduciendo,
                    permisoCamara: _permisoCamara,
                    onPressed: _toggleTraduccion,
                  ),
                  const SizedBox(height: 20),
                  _PanelTraduccion(
                    texto: _textoTraduccion,
                    confianza: _confianza,
                    sonidoActivado: _sonidoActivado,
                    onToggleSonido: _toggleSonido,
                  ),
                  const SizedBox(height: 20),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// =========================================================================
// WIDGETS
// =========================================================================

class _CameraPreview extends StatelessWidget {
  final bool activo;
  final bool isConnected;
  final CameraController? controller;

  const _CameraPreview({
    required this.activo,
    required this.isConnected,
    this.controller,
  });

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 1.05,
      child: Container(
        decoration: BoxDecoration(
          color: Colors.black,
          borderRadius: BorderRadius.circular(20),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(20),
          child: Stack(
            children: [
              if (activo && controller != null && controller!.value.isInitialized)
                CameraPreview(controller!),
              if (!activo || controller == null || !controller!.value.isInitialized)
                Container(
                  color: Colors.black.withOpacity(0.6),
                  child: Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          activo && isConnected ? Icons.videocam : Icons.videocam_off_outlined,
                          size: 56,
                          color: activo && isConnected
                              ? const Color(0xFF3FE08A)
                              : const Color(0xFF6B6B76),
                        ),
                        if (!activo) ...[
                          const SizedBox(height: 14),
                          const Text(
                            'Cámara apagada',
                            style: TextStyle(
                              color: Color(0xFF9B9BA5),
                              fontSize: 15,
                            ),
                          ),
                        ],
                        if (activo && !isConnected) ...[
                          const SizedBox(height: 14),
                          const Text(
                            'Conectando al servidor...',
                            style: TextStyle(
                              color: Colors.white70,
                              fontSize: 14,
                            ),
                          ),
                          const SizedBox(height: 8),
                          const SizedBox(
                            width: 24,
                            height: 24,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          ),
                        ],
                        if (activo && isConnected && (controller == null || !controller!.value.isInitialized)) ...[
                          const SizedBox(height: 14),
                          const Text(
                            'Esperando video...',
                            style: TextStyle(
                              color: Colors.white70,
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
              Positioned(
                top: 14,
                right: 14,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: activo && isConnected
                        ? const Color(0xFF3FB27F)
                        : const Color(0xFFE0475B),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Text(
                    activo && isConnected ? 'ACTIVO' : 'INACTIVO',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.3,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _BotonTraduccion extends StatelessWidget {
  final bool activo;
  final bool permisoCamara;
  final VoidCallback onPressed;

  const _BotonTraduccion({
    required this.activo,
    required this.permisoCamara,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    // Sin permiso, el botón se ve gris/apagado (pero sigue siendo
    // tocable: al tocarlo se muestra el aviso pidiendo el permiso,
    // en vez de simplemente no hacer nada).
    final sinPermiso = !activo && !permisoCamara;

    return SizedBox(
      height: 52,
      child: ElevatedButton.icon(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: activo
              ? const Color(0xFFE85B5B)
              : sinPermiso
                  ? const Color(0xFFB7BAC4)
                  : const Color(0xFF3E5C8A),
          foregroundColor: Colors.white,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
        icon: Icon(
          activo
              ? Icons.stop
              : sinPermiso
                  ? Icons.lock_outline
                  : Icons.play_arrow,
          size: 22,
        ),
        label: Text(
          activo ? 'Detener traducción' : 'Iniciar traducción',
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }
}

class _PanelTraduccion extends StatelessWidget {
  final String texto;
  final double confianza;
  final bool sonidoActivado;
  final VoidCallback onToggleSonido;

  const _PanelTraduccion({
    required this.texto,
    required this.confianza,
    required this.sonidoActivado,
    required this.onToggleSonido,
  });

  @override
  Widget build(BuildContext context) {
    // El texto por defecto ("Presiona iniciar para traducir") se ve más
    // apagado (opacidad baja) para que no compita visualmente con una
    // palabra real ya reconocida. En cuanto hay una traducción real, se
    // muestra con opacidad completa.
    final esPlaceholder = texto == 'Presiona iniciar para traducir';

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: const Color(0xFFEAEBF5),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'TRADUCCIÓN EN TIEMPO REAL',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  color: Color(0xFF3E5C8A),
                  letterSpacing: 0.3,
                ),
              ),
              // Tocable: activa/desactiva que la app diga la palabra en
              // voz alta. Tachado cuando está desactivado.
              InkWell(
                borderRadius: BorderRadius.circular(20),
                onTap: onToggleSonido,
                child: Padding(
                  padding: const EdgeInsets.all(4),
                  child: Icon(
                    sonidoActivado ? Icons.volume_up : Icons.volume_off,
                    color: const Color(0xFF3E5C8A),
                    size: 20,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Center(
            child: Column(
              children: [
                Opacity(
                  opacity: esPlaceholder ? 0.45 : 1.0,
                  child: Text(
                    texto,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: esPlaceholder ? 15 : 24,
                      fontWeight:
                          esPlaceholder ? FontWeight.w400 : FontWeight.w600,
                      color: const Color(0xFF1C1C28),
                    ),
                  ),
                ),
                if (confianza > 0) ...[
                  const SizedBox(height: 8),
                  Container(
                    height: 4,
                    width: 120,
                    decoration: BoxDecoration(
                      color: Colors.grey[300],
                      borderRadius: BorderRadius.circular(2),
                    ),
                    child: FractionallySizedBox(
                      widthFactor: confianza,
                      child: Container(
                        decoration: BoxDecoration(
                          color: confianza > 0.7 ? Colors.green : Colors.orange,
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${(confianza * 100).toStringAsFixed(0)}%',
                    style: const TextStyle(
                      fontSize: 12,
                      color: Color(0xFF7A7A85),
                    ),
                  ),
                ],
              ],
            ),
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}
