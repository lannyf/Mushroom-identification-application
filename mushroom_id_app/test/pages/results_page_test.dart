import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:get/get.dart';
import 'package:mushroom_identification/pages/results_page.dart';
import 'package:mushroom_identification/providers/history_provider.dart';
import 'package:mushroom_identification/providers/identification_provider.dart';
import 'package:mushroom_identification/providers/language_provider.dart';
import 'package:mushroom_identification/utils/app_translations.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

void main() {
  setUpAll(() {
    sqfliteFfiInit();
    databaseFactory = databaseFactoryFfi;
  });

  group('ResultsPage', () {
    late IdentificationProvider idProvider;

    setUp(() async {
      SharedPreferences.setMockInitialValues({});
      Get.reset();
      Get.put(LanguageProvider());
      Get.put(HistoryProvider());
      idProvider = Get.put(IdentificationProvider());
      await Future.delayed(Duration.zero);
    });

    tearDown(() {
      Get.reset();
    });

    Future<void> pumpResultsPage(
      WidgetTester tester, {
      Widget home = const ResultsPage(),
    }) async {
      await tester.pumpWidget(
        GetMaterialApp(
          translations: AppTranslations(),
          locale: const Locale('en', 'US'),
          fallbackLocale: const Locale('en', 'US'),
          home: home,
        ),
      );
      await tester.pump();
    }

    testWidgets('ResultsPage displays correctly',
        (WidgetTester tester) async {
      await pumpResultsPage(tester);

      expect(find.text('Identification Results'), findsOneWidget);
      expect(find.byType(AppBar), findsOneWidget);
    });

    testWidgets('ResultsPage displays confidence indicator',
        (WidgetTester tester) async {
      await pumpResultsPage(tester);

      expect(find.byType(CircularProgressIndicator), findsOneWidget);
    });

    testWidgets('ResultsPage displays confidence percentage',
        (WidgetTester tester) async {
      await pumpResultsPage(tester);

      // Should display confidence percentage
      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              (widget.data?.contains('%') == true ||
                  widget.data?.contains('85') == true),
        ),
        findsWidgets,
      );
    });

    testWidgets('ResultsPage displays predictions section',
        (WidgetTester tester) async {
      await pumpResultsPage(tester);

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Top Predictions') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('ResultsPage displays method breakdown',
        (WidgetTester tester) async {
      await pumpResultsPage(tester);

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              (widget.data?.contains('Image') == true ||
                  widget.data?.contains('Trait') == true ||
                  widget.data?.contains('LLM') == true),
        ),
        findsWidgets,
      );
    });

    testWidgets('ResultsPage displays lookalike warnings',
        (WidgetTester tester) async {
      await pumpResultsPage(tester);

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Lookalike') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('ResultsPage displays safety warning',
        (WidgetTester tester) async {
      await pumpResultsPage(tester);

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Safety') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('ResultsPage has action buttons',
        (WidgetTester tester) async {
      await pumpResultsPage(tester);

      // AppBar has bookmark (save) and share icons; bottom has Share icon
      expect(find.byIcon(Icons.bookmark), findsWidgets);
      expect(find.byIcon(Icons.share), findsWidgets);
    });

    testWidgets('ResultsPage displays disclaimer',
        (WidgetTester tester) async {
      await pumpResultsPage(tester);

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Important Safety Notice') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('ResultsPage shows notice when tree traversal was skipped',
        (WidgetTester tester) async {
      await pumpResultsPage(
        tester,
        home: Builder(
          builder: (context) => Scaffold(
            body: Center(
              child: ElevatedButton(
                onPressed: () => Get.to(
                  () => const ResultsPage(),
                  arguments: {
                    'results': {
                      'final_recommendation': {
                        'swedish_name': 'Flugsvamp',
                        'english_name': 'Fly Agaric',
                        'scientific_name': 'Amanita muscaria',
                        'overall_confidence': 0.93,
                        'confidence_breakdown': {
                          'image_analysis': 0.93,
                          'tree_traversal': 0.0,
                          'trait_match': 0.0,
                        },
                        'reasoning': 'ML precheck fallback.',
                      },
                      'ml_alternatives': const [],
                      'exchangeable_species': const [],
                      'safety_warnings': const [],
                      'verdict': 'inedible',
                      'method_agreement': 'partial',
                    },
                    'step2Result': {
                      'status': 'conclusion',
                      'species': 'Flugsvamp',
                      'message': 'Skipped species tree traversal because Fly Agaric cannot be confirmed exactly by key.xml.',
                      'tree_compatibility': {
                        'tree_policy': 'skip_tree',
                      },
                    },
                  },
                ),
                child: const Text('open'),
              ),
            ),
          ),
        ),
      );

      await tester.tap(find.text('open'));
      await tester.pumpAndSettle();

      expect(
        find.text('Species tree traversal was not possible for this mushroom.'),
        findsOneWidget,
      );
      expect(
        find.textContaining('Skipped species tree traversal'),
        findsOneWidget,
      );
    });
  });
}
