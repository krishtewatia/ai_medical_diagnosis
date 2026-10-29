# Image upload samples

Downloaded on 29 October 2026. These JPEGs are ready to select in the live site's image upload forms. They are public educational examples, not a clinical validation set. The source descriptions below are not guaranteed model outputs. Both samples in each folder show a condition described by the source; there are no healthy control images in this small set.

## Upload instructions

1. Sign in to the app and open **Diseases**.
2. For **Pneumonia Chest X-ray Screening**, upload either JPEG in `pneumonia/`.
3. For **Brain Tumor MRI Screening**, upload either JPEG in `brain_tumor/`.
4. Run the screening, then open History to check the saved result and PDF report.

All files are below the backend's 15 MB upload limit. Keep the downloaded resolution; the backend performs its own resizing to 224 x 224. First inference after a free Render service wakes up can take longer while the model loads. Tabular models use form inputs and do not need image files.

## Sources and reuse terms

| Local file | Source description | Credit and license | Source page |
| --- | --- | --- | --- |
| `pneumonia/cdc_pneumonia_chest_xray.jpg` | AP chest radiograph described as pneumonia with right upper lobe shadowing | CDC Public Health Image Library; U.S. federal government public domain | [Pneumonia x ray.jpg](https://commons.wikimedia.org/wiki/File:Pneumonia_x_ray.jpg) |
| `pneumonia/lobar_pneumonia_chest_xray.jpg` | Chest radiograph described as right middle lobar pneumonia | Mikael Haggstrom, M.D.; [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | [X-ray of lobar pneumonia.jpg](https://commons.wikimedia.org/wiki/File:X-ray_of_lobar_pneumonia.jpg) |
| `brain_tumor/glioblastoma_mri_afip_00405558.jpg` | MRI example of glioblastoma with ring-like contrast enhancement | Armed Forces Institute of Pathology; U.S. federal government public domain | [AFIP-00405558-Glioblastoma-Radiology.jpg](https://commons.wikimedia.org/wiki/File:AFIP-00405558-Glioblastoma-Radiology.jpg) |
| `brain_tumor/recurrent_glioblastoma_mri_afip_00405589.jpg` | MRI example described as recurrent glioblastoma; source notes enhancement can also reflect radiation injury | Armed Forces Institute of Pathology; U.S. federal government public domain | [AFIP-00405589-Glioblastoma-Radiology.jpg](https://commons.wikimedia.org/wiki/File:AFIP-00405589-Glioblastoma-Radiology.jpg) |

Original image bytes are retained without edits. File names were changed for clarity. Wikimedia Commons hosts the originals and their individual license records; these image licenses are independent of the application's source code license. Do not use the screening output as a diagnosis.
