from pathlib import Path
from playwright.sync_api import sync_playwright


class DetectorVisualCheck:
    def run(self):
        directory = Path(__file__).resolve().parent
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                headless=True,
            )
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            for width in (1024, 736, 360):
                for theme in ('light', 'dark'):
                    page.set_viewport_size({'width': width, 'height': 1500})
                    page.emulate_media(color_scheme=theme)
                    page.goto((directory / 'wide-detectors-preview.html').as_uri())
                    frame = page.frames[1]
                    frame.wait_for_selector('#wd-title-0')
                    frame.wait_for_function("document.getElementById('wd-title-0').textContent.includes('12/49')")
                    assert '9/196' in frame.locator('#wd-title-1').inner_text()
                    assert '4/784' in frame.locator('#wd-title-2').inner_text()
                    assert '270.0°' in frame.locator('#wd-angular-status').inner_text()
                    assert '8/448' in frame.locator('#wd-angular-status').inner_text()
                    assert frame.locator('svg').count() == 7
                    for index, count in enumerate((360, 72, 12, 4)):
                        plot = frame.locator(f'#wd-angular-{index}')
                        assert plot.locator('[data-detector]').count() == count
                        assert plot.locator('[data-active="true"]').count() == 2
                        assert 'перекрытие 25%' in plot.inner_text()
                    assert frame.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    page.screenshot(path=str(directory / f'detectors-{width}-{theme}.png'), full_page=True)
                    frame.locator('#wd-angle').fill('0')
                    frame.locator('#wd-angle').dispatch_event('input')
                    assert '180.0°' in frame.locator('#wd-angular-status').inner_text()
                    assert '(1.500; 1.000)' in frame.locator('#wd-spatial-status').inner_text()
                    frame.locator('#wd-angle').fill('45')
                    frame.locator('#wd-angle').dispatch_event('input')
                    assert '225.0°' in frame.locator('#wd-angular-status').inner_text()
                    assert frame.locator('#wd-angular-3 [data-active="true"]').count() == 1
                    print(width, theme, 'OK')
            assert not errors, errors
            browser.close()


if __name__ == '__main__':
    DetectorVisualCheck().run()
