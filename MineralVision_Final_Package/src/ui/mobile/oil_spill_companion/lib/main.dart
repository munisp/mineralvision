import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

void main() => runApp(const OilSpillCompanionApp());

class OilSpillCompanionApp extends StatelessWidget {
  const OilSpillCompanionApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'MineralVision Field Ops',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0891B2), brightness: Brightness.light),
          useMaterial3: true,
        ),
        home: const BootstrapPage(),
      );
}

class BootstrapPage extends StatefulWidget {
  const BootstrapPage({super.key});

  @override
  State<BootstrapPage> createState() => _BootstrapPageState();
}

class _BootstrapPageState extends State<BootstrapPage> {
  final _storage = const FlutterSecureStorage();
  bool _loading = true;
  String? _apiUrl;
  String? _token;

  @override
  void initState() {
    super.initState();
    _loadConfiguration();
  }

  Future<void> _loadConfiguration() async {
    final values = await Future.wait([_storage.read(key: 'apiUrl'), _storage.read(key: 'accessToken')]);
    if (!mounted) return;
    setState(() {
      _apiUrl = values[0];
      _token = values[1];
      _loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) return const Scaffold(body: Center(child: CircularProgressIndicator()));
    if (_apiUrl == null || _token == null) {
      return SettingsPage(onSaved: _loadConfiguration);
    }
    return OperationsHome(api: OilSpillApi(baseUrl: _apiUrl!, token: _token!), onSettings: _loadConfiguration);
  }
}

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key, required this.onSaved});
  final Future<void> Function() onSaved;

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  final _storage = const FlutterSecureStorage();
  final _url = TextEditingController(text: 'https://your-api.example');
  final _token = TextEditingController();
  bool _saving = false;

  Future<void> _save() async {
    if (_url.text.trim().isEmpty || _token.text.trim().isEmpty) return;
    setState(() => _saving = true);
    await _storage.write(key: 'apiUrl', value: _url.text.trim().replaceFirst(RegExp(r'/$'), ''));
    await _storage.write(key: 'accessToken', value: _token.text.trim());
    await widget.onSaved();
    if (mounted) setState(() => _saving = false);
  }

  @override
  void dispose() {
    _url.dispose();
    _token.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Connect MineralVision')),
        body: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: ListView(
              children: [
                const Icon(Icons.shield_outlined, size: 52),
                const SizedBox(height: 16),
                Text('Secure field companion', style: Theme.of(context).textTheme.headlineSmall, textAlign: TextAlign.center),
                const SizedBox(height: 8),
                const Text('Enter the HTTPS API URL and a short-lived operator access token. The token is retained only in the device secure store.'),
                const SizedBox(height: 24),
                TextField(controller: _url, keyboardType: TextInputType.url, decoration: const InputDecoration(labelText: 'API base URL', border: OutlineInputBorder())),
                const SizedBox(height: 16),
                TextField(controller: _token, obscureText: true, decoration: const InputDecoration(labelText: 'Access token', border: OutlineInputBorder())),
                const SizedBox(height: 20),
                FilledButton.icon(onPressed: _saving ? null : _save, icon: const Icon(Icons.lock_open_outlined), label: Text(_saving ? 'Saving…' : 'Connect securely')),
                const SizedBox(height: 20),
                const Text('This app does not launch aircraft, notify authorities, or authorize cleanup. It captures evidence and records review decisions only.', style: TextStyle(fontSize: 12)),
              ],
            ),
          ),
        ),
      );
}

class OperationsHome extends StatefulWidget {
  const OperationsHome({super.key, required this.api, required this.onSettings});
  final OilSpillApi api;
  final Future<void> Function() onSettings;

  @override
  State<OperationsHome> createState() => _OperationsHomeState();
}

class _OperationsHomeState extends State<OperationsHome> {
  int _index = 0;
  Map<String, dynamic>? _summary;
  List<Map<String, dynamic>> _incidents = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    setState(() { _loading = true; _error = null; });
    try {
      final results = await Future.wait([widget.api.summary(), widget.api.incidents(reviewStatus: 'pending_review')]);
      if (!mounted) return;
      setState(() { _summary = results[0] as Map<String, dynamic>; _incidents = results[1] as List<Map<String, dynamic>>; });
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final pages = [
      OperationsDashboard(summary: _summary, incidents: _incidents, loading: _loading, error: _error, onRefresh: _refresh),
      IncidentList(incidents: _incidents, onReview: _openReview),
      CaptureEvidencePage(api: widget.api, onSubmitted: _refresh),
    ];
    return Scaffold(
      appBar: AppBar(
        title: const Text('Oil Spill Field Ops'),
        actions: [IconButton(onPressed: () => widget.onSettings(), icon: const Icon(Icons.settings_outlined), tooltip: 'Connection settings')],
      ),
      body: SafeArea(child: pages[_index]),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (value) => setState(() => _index = value),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.dashboard_outlined), selectedIcon: Icon(Icons.dashboard), label: 'Triage'),
          NavigationDestination(icon: Icon(Icons.fact_check_outlined), selectedIcon: Icon(Icons.fact_check), label: 'Review'),
          NavigationDestination(icon: Icon(Icons.add_a_photo_outlined), selectedIcon: Icon(Icons.add_a_photo), label: 'Capture'),
        ],
      ),
    );
  }

  Future<void> _openReview(Map<String, dynamic> incident) async {
    final decision = await showModalBottomSheet<String>(
      context: context,
      showDragHandle: true,
      builder: (context) => Padding(
        padding: const EdgeInsets.all(24),
        child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Text('Review incident', style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 8),
          Text('Source: ${incident['source']} · severity: ${incident['severity']}'),
          const SizedBox(height: 16),
          for (final status in const ['confirmed', 'needs_resurvey', 'false_positive'])
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: FilledButton.tonal(onPressed: () => Navigator.pop(context, status), child: Text(status.replaceAll('_', ' '))),
            ),
        ]),
      ),
    );
    if (decision == null) return;
    try {
      await widget.api.review(incident['incident_id'] as String, status: decision, reviewer: 'mobile_operator');
      await _refresh();
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Review recorded. No operational action was dispatched.')));
    } catch (_) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Review was not saved. Check connection and token.')));
    }
  }
}

class OperationsDashboard extends StatelessWidget {
  const OperationsDashboard({super.key, required this.summary, required this.incidents, required this.loading, required this.error, required this.onRefresh});
  final Map<String, dynamic>? summary;
  final List<Map<String, dynamic>> incidents;
  final bool loading;
  final String? error;
  final Future<void> Function() onRefresh;

  @override
  Widget build(BuildContext context) => RefreshIndicator(
        onRefresh: onRefresh,
        child: ListView(padding: const EdgeInsets.all(16), children: [
          const Text('Decision support only', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.teal)),
          const SizedBox(height: 4),
          const Text('Review evidence and coordinate through approved incident-response procedures.'),
          const SizedBox(height: 16),
          if (loading) const Center(child: Padding(padding: EdgeInsets.all(32), child: CircularProgressIndicator()))
          else if (error != null) Card(child: Padding(padding: const EdgeInsets.all(16), child: Text('Unable to load operations data: $error')))
          else ...[
            GridView.count(
              crossAxisCount: 2,
              childAspectRatio: 1.55,
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              children: [
                MetricCard(label: 'Pending review', value: '${summary?['pending_review'] ?? 0}', icon: Icons.manage_search),
                MetricCard(label: 'High / critical', value: '${summary?['high_or_critical'] ?? 0}', icon: Icons.warning_amber_rounded),
                MetricCard(label: 'Confirmed', value: '${summary?['confirmed'] ?? 0}', icon: Icons.verified_outlined),
                MetricCard(label: 'Approved models', value: '${summary?['approved_models'] ?? 0}', icon: Icons.security_outlined),
              ],
            ),
            const SizedBox(height: 20),
            Text('Priority queue', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            if (incidents.isEmpty) const Card(child: Padding(padding: EdgeInsets.all(16), child: Text('No pending incidents. Pull to refresh.')))
            else ...incidents.take(5).map((incident) => Card(child: ListTile(
              leading: const Icon(Icons.location_searching),
              title: Text('${incident['source']} · ${incident['severity']}'),
              subtitle: Text('Confidence: ${incident['confidence'] == null ? '—' : '${((incident['confidence'] as num) * 100).round()}%'}'),
              trailing: const Icon(Icons.chevron_right),
            ))),
          ],
        ]),
      );
}

class MetricCard extends StatelessWidget {
  const MetricCard({super.key, required this.label, required this.value, required this.icon});
  final String label;
  final String value;
  final IconData icon;
  @override
  Widget build(BuildContext context) => Card(child: Padding(padding: const EdgeInsets.all(14), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Icon(icon, color: Theme.of(context).colorScheme.primary), const Spacer(), Text(value, style: Theme.of(context).textTheme.headlineSmall), Text(label, style: Theme.of(context).textTheme.labelMedium)])));
}

class IncidentList extends StatelessWidget {
  const IncidentList({super.key, required this.incidents, required this.onReview});
  final List<Map<String, dynamic>> incidents;
  final Future<void> Function(Map<String, dynamic>) onReview;
  @override
  Widget build(BuildContext context) => ListView(
        padding: const EdgeInsets.all(12),
        children: [
          const Padding(padding: EdgeInsets.all(8), child: Text('Pending-review queue', style: TextStyle(fontWeight: FontWeight.bold))),
          if (incidents.isEmpty) const Padding(padding: EdgeInsets.all(24), child: Text('No pending incidents.')),
          ...incidents.map((incident) => Card(child: ListTile(
            title: Text('${incident['source']} evidence'),
            subtitle: Text('${incident['severity']} · ${incident['review_status']}'),
            trailing: const Icon(Icons.rate_review_outlined),
            onTap: () => onReview(incident),
          ))),
        ],
      );
}

class CaptureEvidencePage extends StatefulWidget {
  const CaptureEvidencePage({super.key, required this.api, required this.onSubmitted});
  final OilSpillApi api;
  final Future<void> Function() onSubmitted;
  @override
  State<CaptureEvidencePage> createState() => _CaptureEvidencePageState();
}

class _CaptureEvidencePageState extends State<CaptureEvidencePage> {
  final _picker = ImagePicker();
  XFile? _image;
  String _source = 'drone_rgb';
  bool _submitting = false;

  Future<void> _capture() async {
    final image = await _picker.pickImage(source: ImageSource.camera, imageQuality: 90);
    if (image != null && mounted) setState(() => _image = image);
  }

  Future<void> _submit() async {
    if (_image == null) return;
    setState(() => _submitting = true);
    try {
      await widget.api.analyzeImage(File(_image!.path), source: _source);
      await widget.onSubmitted();
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Image sent for controlled model analysis and human review.')));
    } catch (error) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Raw-image submission failed. The server may not have an approved model configured.')));
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  Widget build(BuildContext context) => ListView(padding: const EdgeInsets.all(16), children: [
        const Text('Capture field evidence', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        const Text('Raw image analysis is enabled only when the server has a hash-verified, registered, and approved segmentation model. Capture does not dispatch operational action.'),
        const SizedBox(height: 18),
        DropdownButtonFormField(value: _source, decoration: const InputDecoration(labelText: 'Observation source', border: OutlineInputBorder()), items: const [DropdownMenuItem(value: 'drone_rgb', child: Text('Drone RGB')), DropdownMenuItem(value: 'satellite_rgb', child: Text('Satellite RGB')), DropdownMenuItem(value: 'fluorosensor', child: Text('Fluorosensor'))], onChanged: (value) => setState(() => _source = value!)),
        const SizedBox(height: 16),
        OutlinedButton.icon(onPressed: _capture, icon: const Icon(Icons.photo_camera_outlined), label: Text(_image == null ? 'Capture image' : 'Retake image')),
        if (_image != null) Padding(padding: const EdgeInsets.only(top: 12), child: Text('Selected: ${_image!.name}')),
        const SizedBox(height: 16),
        FilledButton.icon(onPressed: _image == null || _submitting ? null : _submit, icon: const Icon(Icons.cloud_upload_outlined), label: Text(_submitting ? 'Submitting…' : 'Submit to controlled analysis')),
      ]);
}

class OilSpillApi {
  OilSpillApi({required this.baseUrl, required this.token});
  final String baseUrl;
  final String token;

  Map<String, String> get _headers => {'Authorization': 'Bearer $token', 'Accept': 'application/json'};

  Future<Map<String, dynamic>> summary() async => _object(await http.get(Uri.parse('$baseUrl/api/oil-spill/operations/summary'), headers: _headers));

  Future<List<Map<String, dynamic>>> incidents({String? reviewStatus}) async {
    final suffix = reviewStatus == null ? '' : '?review_status=$reviewStatus';
    final response = await http.get(Uri.parse('$baseUrl/api/oil-spill/incidents$suffix'), headers: _headers);
    if (response.statusCode < 200 || response.statusCode >= 300) throw HttpException('Incident list failed (${response.statusCode})');
    return (jsonDecode(response.body) as List<dynamic>).cast<Map<String, dynamic>>();
  }

  Future<void> review(String incidentId, {required String status, required String reviewer}) async {
    final response = await http.patch(Uri.parse('$baseUrl/api/oil-spill/incidents/$incidentId/review'), headers: {..._headers, 'Content-Type': 'application/json'}, body: jsonEncode({'status': status, 'reviewer': reviewer}));
    if (response.statusCode < 200 || response.statusCode >= 300) throw HttpException('Review failed (${response.statusCode})');
  }

  Future<void> analyzeImage(File image, {required String source}) async {
    final request = http.MultipartRequest('POST', Uri.parse('$baseUrl/api/oil-spill/analyze/image'));
    request.headers.addAll(_headers);
    request.fields['source'] = source;
    request.files.add(await http.MultipartFile.fromPath('image', image.path));
    final response = await request.send();
    if (response.statusCode < 200 || response.statusCode >= 300) throw HttpException('Image analysis failed (${response.statusCode})');
  }

  Map<String, dynamic> _object(http.Response response) {
    if (response.statusCode < 200 || response.statusCode >= 300) throw HttpException('API request failed (${response.statusCode})');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }
}
