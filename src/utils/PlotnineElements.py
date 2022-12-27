
from PIL import Image
import plotnine as gg
from typing import List, Optional, Union

blank = gg.element_blank()


class PlotnineElements:
    """
    Class of static methods to modify plotnine plots
    """

    colour2 = ["#1e8ef0", "#ff4f00"]
    colour3 = ["#1e8ef0", "#16e276", "#ff4f00"]
    colour4 = ["#1e8ef0", "#16e276", "#ff4f00", "#dd1616"]

    @staticmethod
    def text(size: Optional[int] = None,
             colour: Optional[str] = None,
             weight: Optional[str] = None):
        """
        Text size, colour and weight, excluded arguments will take default values
        """
        
        return gg.theme(text=gg.element_text(size=size, color=colour, weight=weight))

    @staticmethod
    def labels(title: Optional[str] = None, 
               x: Optional[str] = None, 
               y: Optional[str] = None,
               title_size: Optional[int] = None,
               x_size: Optional[int] = None,
               y_size: Optional[int] = None):
        """
        Labels for plotnine, excluded arguments will take default values, label arguments with '' will be blank
        """
        # This can have weird effects with gg.flip_coords()

        # Return a list of the functions to be applied
        return [gg.ggtitle(title) if title else None,
                gg.xlab(x) if x else None,
                gg.ylab(y) if y else None,
                gg.theme(plot_title=(gg.element_text(size=title_size) if title_size else (blank if title == "" else None)),
                         axis_title_x=(gg.element_text(size=x_size) if x_size else (blank if x == "" else None)),
                         axis_title_y=(gg.element_text(size=y_size) if y_size else (blank if y == "" else None)))]
    
    @staticmethod
    def axis_limits(x: Optional[tuple[Union[int, float, None], Union[int, float, None]]] = (None, None),
                    x_expand: Optional[tuple[Union[int, float], Union[int, float]]] = None,
                    y: Optional[tuple[Union[int, float, None], Union[int, float, None]]] = (None, None),
                    y_expand: Optional[tuple[Union[int, float], Union[int, float]]] = None,
                    ):
        """
        Axis limits using scale_x/y_continuous, excluded arguments will take default values
        """

        return [gg.scale_x_continuous(limits=x, expand=x_expand),
                gg.scale_y_continuous(limits=y, expand=y_expand)]

    @staticmethod
    def remove_ticks(x_minor: bool = False,
                     x_major: bool = False,
                     y_minor: bool = False,
                     y_major: bool = False,
                     major: bool = False,
                     minor: bool = False,
                     x: bool = False,
                     y: bool = False):
        """
        Remove axis ticks as specified by arguments, True to remove
        """

        return gg.theme(axis_ticks_minor_x=(blank if x_minor or minor or x else None),
                        axis_ticks_minor_y=(blank if y_minor or minor or y else None),
                        axis_ticks_major_x=(blank if x_major or major or x else None),
                        axis_ticks_major_y=(blank if y_major or major or y else None))
        
    @staticmethod
    def remove_grid(x_minor: bool = False,
                    x_major: bool = False,
                    y_minor: bool = False,
                    y_major: bool = False,
                    major: bool = False,
                    minor: bool = False,
                    x: bool = False,
                    y: bool = False):
        """
        Remove axis ticks as specified by arguments, True to remove
        """

        return gg.theme(panel_grid_minor_x=(blank if x_minor or minor or x else None),
                        panel_grid_minor_y=(blank if y_minor or minor or y else None),
                        panel_grid_major_x=(blank if x_major or major or x else None),
                        panel_grid_major_y=(blank if y_major or major or y else None))

    @staticmethod
    def background_colour(colour: Optional[str] = None,
                          plot_colour: Optional[str] = None,
                          panel_colour: Optional[str] = None):
        """
        Specify background colour.  Set both with colour or separately with plot and panel.  Plot is the entire area
        """

        if colour is not None:
            return gg.theme(plot_background = gg.element_rect(fill=colour, colour=colour),
                            panel_background = gg.element_rect(fill=colour),
                            legend_background = gg.element_rect(fill=colour))
        elif plot_colour is not None and panel_colour is not None:
            return gg.theme(plot_background = gg.element_rect(fill=plot_colour),
                            panel_background = gg.element_rect(fill=panel_colour))
        else:
            return None
            

class ImageCombine:
    """
    Utility class just for combining images horizontally or vertically.
    """

    @staticmethod
    def combine_plots_vertical(images: List, base_filepath: str = "./") -> Image.Image:
        """
        Combine given images into a single image
        """
        images = [Image.open(base_filepath + x) for x in images]
        widths, heights = zip(*(i.size for i in images))

        total_width = max(widths)
        total_height = sum(heights)

        # Resize images to the same width, the max width of the images
        images = [im.resize((total_width, im.size[1])) for im in images]

        new_image = Image.new('RGB', (total_width, total_height))
        x_offset = 0
        for image in images:
            new_image.paste(image, (0, x_offset))
            x_offset += image.size[1]

        return new_image
    
    @staticmethod
    def combine_plots_horizontal(images: List, base_filepath: str = "./") -> Image.Image:
        """
        Combine given images into a single image
        """
        images = [Image.open(base_filepath + x) for x in images]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        total_height = max(heights)

        # Resize images to the same width, the max width of the images
        images = [im.resize((im.size[0], total_height)) for im in images]

        new_image = Image.new('RGB', (total_width, total_height))
        y_offset = 0
        for image in images:
            new_image.paste(image, (y_offset, 0))
            y_offset += image.size[0]

        return new_image
    
    @staticmethod
    def save_image(image: Image.Image, name: str, filepath: str = "./"):
        filepath = filepath + name + ".png"
        image.save(fp=filepath, format="png")
